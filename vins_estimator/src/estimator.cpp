#include "estimator.h"
#include "feature_manager.h"
#include <thread>

std::mutex mtx_GF ;

int number = 0;
double time_used = 0.0;
Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //相机和IMU之间的外参
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();//计算图像测量残差VI-C的协方差矩阵P_l^{cj}
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}


void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;   
		//Rs,Ps,Vs存放了相对于前一帧的变换关系(因为一开始不知道在世界坐标系的坐标，以初始帧的位姿作为参考，在与IMU结果对齐的过程中再变换回世界坐标系中)
		//到世界坐标系的变换是在visualInitialAlign函数最后中做的
		//此处不再是预积分，而是普通积分，求得的是在绝对坐标系下的坐标
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();//相对于上一个IMU数据的转角
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;//将IMU的两帧间预积分结果赋值给图像
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        selection_flag = false;
        //注意最新帧的frame_count的标号为WINDOW_SIZE,这意味着滑窗的帧数为WINDOW_SIZE+1帧图像。
        //这也可以在接下来的initialStructure()中relativePose函数中得到验证
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0]; 
				selection_flag = true;
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
		selectFrame = imageframe;
        solveOdometry();
		time_used += t_solve.toc();
		number++;
        ROS_INFO("used time: %fms, number: %d", time_used, number);
		//检测到跟踪失败，系统转到初始化阶段
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
		//从第二帧图像开始
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global window sfm
    Quaterniond Q[frame_count + 1];//因为frame_count是从0开始计时的，所以数组实际长度要+1
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;//SFMFeature里面记录了每个路标点在所有观测到该点的帧中的信息
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;//观测到路标点的帧的标号
            Vector3d pts_j = it_per_frame.point;
			//将某个路标点在各个观测帧中的x,y坐标值存入
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
	//指的从后一帧坐标系变换到前一帧坐标系的变换矩阵
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;//窗口中最早满足与当前帧视差关系的帧的标号
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
	//三角化出窗口内所有特征点的三维坐标，并对窗口中各个帧的位姿进行优化
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    //因为有可能一个窗口中不能完成优化，前面已经有很多帧，construct()函数中没有对这些帧进行处理，此处对这些帧的位姿也进行了处理 
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;//IMU速度，重力向量，尺度参数
    //solve scale V-B中所有的内容
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state 对应于V-B(4)completing Initialization
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
		//将窗口中所有的帧都设置为关键帧
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    //此处将所有初始化过程中恢复出来的深度为正值的路标点的深度重新初始化为-1,目的是为了接下来triangulate函数中对所有这些点重新三角化
    //因为triangulate函数中判断条件就是if (it_per_id.estimated_depth > 0)
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);//构造一个大小为1的块，元素是向量的最后一个元素
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();//根据重力计算初始帧坐标系相对于世界坐标系的旋转矩阵
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;//得到世界坐标系中重力向量的表示
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
	//将在第一帧坐标系中的表示转换为在世界坐标系中的表示
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

//从窗口中找到最早满足与当前帧视差条件的关键帧，然后求解出这两帧间的相对位姿
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //由上面的英文注释我们可以知道最新帧的frame_count=WINDOW_SIZE
	for (int i = 0; i < WINDOW_SIZE; i++)//
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);//之前帧与最后一帧中对应的特征点
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
	if (solver_flag == NON_LINEAR && selection_flag == false)
    {
        f_manager.triangulate(Ps, tic, ric);
        optimization();
		ROS_INFO("flag == false!");
    }
    else if (solver_flag == NON_LINEAR && selection_flag == true)
    {
        ROS_INFO("flag == true!");
        f_manager.triangulate(Ps, tic, ric);
		if(!mpVecSelectedBefore.empty())
			mpVecSelectedBefore.clear();
		if(!mpVecSelected.empty())
	    {
	       //如果之前有选择的特征点，且以前选择的特征点在该帧中，则将其预存入数组
           for(auto it : mpVecSelected)
           {
               int feature_id = it.idx;
               for(auto it_result : this->pFrame->points)
               {
                 if(it_result.first == feature_id)
			       mpVecSelectedBefore.push_back(it);  
               }     
		   }		   	
		}
		setSelction_Number(60, 3, 0.001, &selectFrame, mpVecSelected);//&all_image_frame.end()->second, mpVecTest);
        ROS_INFO("Original number: %d, Used number: %d", f_manager.feature.size(), f_manager.feature_selected.size());
		optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

//此处为什么有很多return true被注释掉？这样会不会最后返回了false而导致实际跟踪失败而没有判断出来？
bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}


void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();//
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);//IMU在世界坐标系中的位姿
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);//包括速度和漂移
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);//IMU与相机的位姿
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    //论文中公式22第一项先验误差
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //论文中公式22第二项IMU测量残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;

	//论文中公式22第三项图像测量残差
	//计算的是地图点在不同的图像中的投影误差
    for (auto &it_per_id : selection_flag == false ? f_manager.feature : f_manager.feature_selected)
    //for (auto &it_per_id : f_manager.feature_selected)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
		
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        //遍历地图点在非第一次观测到的帧中的三维坐标
		for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : selection_flag == false ? f_manager.feature : f_manager.feature_selected)
        //for (auto &it_per_id : f_manager.feature_selected)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : selection_flag == false ? f_manager.feature : f_manager.feature_selected)
            //for (auto &it_per_id : f_manager.feature_selected)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            //for循环结束后frame_num == 0变为了frame_num == WINDOW_SIZE-1
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

				//将早于滑窗第一帧之前的帧置位
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;//修改深度
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;//指的在所有帧中的索引值
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;//指的在滑窗内的索引值
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

/**********************************below functions are used for good features selection*******/
//ldmks:is short for landmarks
bool Estimator::setSelction_Number(const size_t num_good_inlier,
                                       const int greedy_mtd,
                                       const double error_bound,
                                       ImageFrame *pFrame, vector<GoodPoint> &mpVec) {

	if (pFrame == NULL )// || mpVec == NULL) 
        return false;

    const int N = pFrame->points.size();

    this->pFrame = pFrame;
    // double random_sample_scale = 6.0; // 16.0; // 20.0; // 12.0; // 10.0; // 4.0; //
    //    double errorBound = 0.1; // 0.05; // 0.01; //
    double random_sample_scale = static_cast<size_t>( float(N) / float(num_good_inlier) * log(1.0 / error_bound) );//对应论文中算法上面的公式

    // Matrix construction for each feature (independently, therefore in parallel)
    runMatrixBuilding(true);

	if(!f_manager.feature_selected.empty())
    {
      f_manager.feature_selected.clear();
	  for(auto it : f_manager.feature)
	  	f_manager.feature_selected.push_back(it);
	}
	else
	{
      for(auto it : f_manager.feature)
	  	f_manager.feature_selected.push_back(it);
	}

    lmkSelectPool.clear();
    // Collect the lmkSelectPool
    for (auto &id_pts : pFrame->points)
    {
		int feature_id = id_pts.first;
        auto it = find_if(f_manager.feature.begin(), f_manager.feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == f_manager.feature.end())
        {
            cout << "didn't find the corresponding feature point" << endl;
			//return false;
        }
        else if (it->feature_id == feature_id && it->selected == false && it->estimated_depth>0 && it->used_num > 1)
        {
            //assert(it->estimated_depth > 0);
			GoodPoint tmpLmk(static_cast<size_t>(it->feature_id), it->feature_per_frame.back().ObsMat);
			lmkSelectPool.push_back( tmpLmk );
        }
    }

	// The sorting process seem to take as much time as the parrallel obs computation
    // Next we will replace the sort part with Max-Basis-Volume approach (O(Nlog(N)) -> O(M2), where M is the number of lmk to use per frame)

    if(!mpVec.empty())
		mpVec.clear();


    //if (greedy_mtd == 1)
        //maxVolSelection_BaselineGreedy(mpVec, num_good_inlier);
    //else 
	if (greedy_mtd == 2) 
	{
	    bool flagSucc = maxVolSelection_LazierGreedy(0, lmkSelectPool.size(), mpVec, num_good_inlier, random_sample_scale);
        if (flagSucc == false) 
		{
            cout << "lazier greedy failed!" << endl;
            return false;
        }
    }
    else if (greedy_mtd == 3)
	{
	    //如果预存入数组的个数超过设置的个数，则从中取相同个数放入最终选择数组退出，否则执行选择
	    if(mpVecSelectedBefore.size() >= num_good_inlier || lmkSelectPool.size() == 0)
	    {
          for(int i = 0; i < num_good_inlier && !mpVecSelectedBefore.empty(); i++)
          {
		  	mpVec.push_back(mpVecSelectedBefore[num_good_inlier-1-i]);
			mpVecSelectedBefore.pop_back();
          }

		  for(int j = 0; j < mpVecSelectedBefore.size(); j++)
		  {
             int feature_id = mpVecSelectedBefore[j].idx;
             auto it = find_if(f_manager.feature.begin(), f_manager.feature.end(), [feature_id](const FeaturePerId &it)
                          {
               return it.feature_id == feature_id;
                          }); 
		     it->selected = false;
		  }

          if(mpVec.empty() )
		  	return false;
		  //将feature的标志位置为true
		  for(auto itv : mpVec)
		  {
           int feature_id = itv.idx;
           auto it = find_if(f_manager.feature.begin(), f_manager.feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          }); 
		   it->selected = true;
		  }
		  return true;
		}

		int final_select_number = num_good_inlier - mpVecSelectedBefore.size();
        // simply call lazier greedy
        bool flagSucc = maxVolAutomatic_LazierGreedy(0, lmkSelectPool.size(), mpVec, final_select_number, random_sample_scale);

		if (flagSucc == false) 
		{
			cout << "lazier greedy failed!" << endl;
            return false;
        }

		for(auto it : mpVecSelectedBefore)
			mpVec.push_back(it);

		//将feature的标志位置为true
		for(auto itv : mpVec)
		{
           int feature_id = itv.idx;
           auto it = find_if(f_manager.feature.begin(), f_manager.feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          }); 
		   it->selected = true;
		}

		//将未被选用的特征点从f_manager中删除
	    for (auto &id_pts : pFrame->points)
		{
			int feature_id = id_pts.first;
			auto it = find_if(f_manager.feature_selected.begin(), f_manager.feature_selected.end(), [feature_id](const FeaturePerId &it)
							  {
				return it.feature_id == feature_id;
							  });
	
			if (it == f_manager.feature_selected.end())
			{
				cout << "didn't find the corresponding feature point" << endl;
				//return false;
			}
			else if (it->feature_id == feature_id && it->selected == false && it->estimated_depth>0 && it->used_num > 1)
			{
				f_manager.feature_selected.erase(it);
			}
		}
		//测试输出是否一致使用
		/*
	    vector<int> original_id;
		vector<int> changed_id;
		for(auto it : f_manager.feature)
     		original_id.push_back(it.feature_id);
		for(auto it : f_manager.feature_selected)
     		changed_id.push_back(it.feature_id);
		std::sort(original_id.begin(), original_id.end());
		std::sort(changed_id.begin(), changed_id.end());
        ROS_INFO("original number:");
		for(auto it : original_id)
		   ROS_INFO("\t %d", it);	
		ROS_INFO("changed number:");
		for(auto it : changed_id)
		   ROS_INFO("\t %d", it);*/
		
    }
    else
        std::cout << "func setSelction_Number: unknown greedy method being called!" << std::endl;
    
    return true;
}

//此函数中特征点的id都统一加了this->pFrame->points.first，是为了防止VINS中frame中的feature id不是从0开始计数 
//如果是从零开始计数也不会对结果的正确性产生影响
bool Estimator::runMatrixBuilding(const bool with_multi_thread) 
{
    int N = 0;

        if (this->pFrame == NULL)
            return false;

        N = this->pFrame->points.size();//mvpMapPoints.size();

        if (with_multi_thread) 
		{
            // original solution, utilizing c++ 11 std thread
            int grainSize = static_cast<double>(N)/static_cast<double>(mNumThreads);

            if (mNumThreads > 1)
                mThreads = new std::thread [mNumThreads-1];

			for (size_t i = 0; i < mNumThreads-1; i++) 
			{
                mThreads[i] = std::thread(&Estimator::batchInfoMat_Frame, this, this->pFrame->points.begin()->first + i*grainSize, this->pFrame->points.begin()->first + (i+1)*grainSize);
            }
            this->batchInfoMat_Frame(this->pFrame->points.begin()->first + (mNumThreads-1)*grainSize, this->pFrame->points.begin()->first + N);

			for (size_t i = 0; i < mNumThreads-1; i++) 
			{
                mThreads[i].join();
            }
            delete [] mThreads;
        }
        else 
		{
            // single-thread matrix construction
            this->batchInfoMat_Frame(this->pFrame->points.begin()->first + 0, this->pFrame->points.begin()->first + N);
        }


    return true;
}

void Estimator::batchInfoMat_Frame(const size_t start_idx, const size_t end_idx)
{
    //不知道是直接删除还是修改，先注释掉了
    //if (this->mKineIdx >= this->kinematic.size())
     //   return ;

    arma::mat H13, H47, H_proj;
    float res_u = 0, res_v = 0, u, v;

    for(size_t i = start_idx; i < end_idx; i++)  
	{
        if (i >= this->pFrame->points.size())
            break ;
		
  
		//FeaturePerFrame* pMP(this->pFrame->points.second[i].second, td); = this->pFrame->mvpMapPoints[i];
        // If the current Map point is not matched:
        // Reset the Obs information
        //int feature_id = this->pFrame->points[i].first;
        auto it = find_if(f_manager.feature.begin(), f_manager.feature.end(), [i](const FeaturePerId &it)
                          {
            return it.feature_id == i;
                          });

        if (it == f_manager.feature.end())
        {
            cout << "didn't find the corresponding feature point" << endl;
			//return;
        }
        else if (it->feature_id == i && it->estimated_depth>0 && it->used_num > 1)
        {
            //assert(it->estimated_depth > 0);

                // skip if the Jacobian has already been built
                if (this->pFrame->mvbJacobBuilt[i] == true)
                    continue ;

                // Feature position
                arma::rowvec Y = arma::zeros<arma::rowvec>(4);
                //cv::Mat Pw = (Mat_<float>(3,1) << it.feature_per_frame.end().point.x()
					                           // << it.feature_per_frame.end().point.y()
					                            //<< it.feature_per_frame.end().point.z();
                Y[0] = it->feature_per_frame.end()->point.x();//Pw.at<float>(0);
                Y[1] = it->feature_per_frame.end()->point.y();//Pw.at<float>(1);
                Y[2] = it->feature_per_frame.end()->point.z();//Pw.at<float>(2);
                Y[3] = 1;

                // Measurement
                arma::rowvec Z = arma::zeros<arma::rowvec>(2);
         
                Z[0] = it->feature_per_frame.end()->uv.x();
                Z[1] = it->feature_per_frame.end()->uv.y();
                //Xv:路标点在世界坐标系下的坐标
                compute_H_subblock_simplied(last_R, last_P, Y.subvec(0, 2), H13, H47, H_proj, u, v);
				res_u = Z[0] - u;
                res_v = Z[1] - v;

                // assemble into H matrix
                arma::mat H_meas = arma::join_horiz(H13, H47);
                arma::mat H_rw;
                reWeightInfoMat( this->pFrame, i, H_meas, res_u, res_v, H_proj, H_rw );
				arma::mat infMat = H_rw.t() * H_rw;

#ifdef MULTI_THREAD_LOCK_ON
                mtx_GF.lock();
#endif
                //pMP->u_proj = u;
                //pMP->v_proj = v;
                //pMP->H_meas = H_meas;
                //pMP->H_proj = H_proj;
                //pMP->ObsMat = infMat;
                // compute observability score
                //pMP->ObsScore = this->pFrame->mvpMatchScore[i];
                this->pFrame->mvbJacobBuilt[i] = true;

#ifdef MULTI_THREAD_LOCK_ON
                mtx_GF.unlock();
#endif
            }
            else 
			{

            }
    } // For: all pMP

}


bool Estimator::maxVolAutomatic_LazierGreedy(const size_t stIdx, const size_t edIdx,
                                                 vector<GoodPoint> &subVec, const size_t mpLimit,
                                                 const double sampleScale) {
                                                 
	if (mpLimit * 2 > edIdx - stIdx) 
	{
        // deletion
        //        std::cout << "Deletion is chosen!" << std::endl;
        return maxVolDeletion_LazierGreedy(stIdx, edIdx, subVec, mpLimit, sampleScale);
    }
    else 
	{
        // addition
        //        std::cout << "Selection is chosen!" << std::endl;
        return maxVolSelection_LazierGreedy(stIdx, edIdx, subVec, mpLimit, sampleScale);
    }
}

bool Estimator::maxVolDeletion_LazierGreedy(const size_t stIdx, const size_t edIdx,
                                                vector<GoodPoint> &subVec, const size_t mpLimit,
                                                const double sampleScale) {
    //    cout << "start lazier greedy from " << stIdx << " to " << edIdx << endl;
    //
	if (lmkSelectPool.size() == 0 )//|| subVec == nullptr) 
        return false;
    if (stIdx >= edIdx || edIdx > lmkSelectPool.size()) 
        return false;

    if(!subVec.empty())
	    subVec.clear();

    if (mpLimit >= edIdx - stIdx) 
	{
        // simply fill in all lmks as selected
        for (size_t i=stIdx; i<edIdx; ++i) 
		{
#ifdef MULTI_THREAD_LOCK_ON
            mtx_GF.lock();
#endif
            lmkSelectPool[i].selected = true;
            

            subVec.push_back(lmkSelectPool[i]);
#ifdef MULTI_THREAD_LOCK_ON
            mtx_GF.unlock();
#endif
        }
        // std::cout << "func maxVolDeletion_LazierGreedy: subset limit higher than input lmk number!" << std::endl;
        return true;
    }

    //    std::cout << "Calling lazier greedy!" << std::endl;
    
    // define the size of random subset
    //    double errorBound = 0.001; // 0.01; //
    size_t szLazierSubset = static_cast<size_t>( double(edIdx - stIdx) / double(mpLimit) * sampleScale );
    //
    // NOTE
    // To obtain stable subset results with random deletion, the size of random pool has to be increased;
    // as a consequence, it actually takes more time than the selection method.
    //
    //    size_t szLazierSubset = static_cast<size_t>( double(mpVec->size()) / double(mpLimit) * log(1.0 / errorBound) * 2);
    arma::mat curMat = arma::eye( size(lmkSelectPool[0].obs_block) ) * 0.00001;

    // iteratively search for the least informative lmk
    // create a query index for fast insert and reject of entries
    arma::mat lmkIdx = arma::mat(1, edIdx - stIdx);
    // create a flag array to avoid duplicate visit
    arma::mat lmkVisited = arma::mat(1, edIdx - stIdx);
    size_t l = 0;
    for (size_t i=stIdx; i<edIdx; ++i) 
	{

#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.lock();
#endif

        curMat = curMat + lmkSelectPool[i].obs_block;
        lmkSelectPool[i].selected = true;

#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.unlock();
#endif
        //
        lmkIdx.at(0, l) = i;
        lmkVisited.at(0, l) = -1;
        //
        ++ l;
    }
    //
    size_t mDelLim = edIdx - stIdx - mpLimit;
    for (size_t i=0; i<mDelLim; ++i) 
	{
        int maxLmk = -1;
        double maxDet = -DBL_MAX;// maximum value of double-point number
        //        std::cout << "func maxVolSubset_Greedy: finding subset idx " << i << std::endl;

        size_t numHit = 0, numRndQue = 0, szActualSubset;
        szActualSubset = szLazierSubset;
        if (lmkIdx.n_cols < szActualSubset)
            szActualSubset = lmkIdx.n_cols;
        //
        while (numHit < szActualSubset) 
		{
            // generate random query index
            //            cout << "random query round " << numHit << endl;
            size_t j;
            numRndQue = 0;
            while (numRndQue < MAX_RANDOM_QUERY_TIME) 
			{
                j = ( std::rand() % lmkIdx.n_cols );
                //                cout << j << " ";
                // check if visited
                if (lmkVisited.at(0, j) < i) {
                    lmkVisited.at(0, j) = i;
                    break ;
                }
                ++ numRndQue;
            }
            if (numRndQue >= MAX_RANDOM_QUERY_TIME)
                break ;

            //            cout << endl;
            int queIdx = lmkIdx.at(0, j);
            ++ numHit;

            if (lmkSelectPool[queIdx].selected == false) 
			{
                // std::cout << "It does not supposed to happen!" << std::endl;
                -- numHit;
                continue ;
            }
        
            arma::cx_double curDet = arma::log_det( curMat - lmkSelectPool[queIdx].obs_block );
            if (curDet.real() > maxDet) 
			{
                maxDet = curDet.real();
                maxLmk = queIdx;
                //                std::cout << "current determinant = " << curDet << std::endl;
            }
        }

        if (maxLmk == -1) 
		{
            std::cout << "func maxVolDeletion_LazierGreedy: early termination!" << std::endl;
            break ;
        }

        // set up the index for columns that are not selected yet
        arma::uvec restCol = arma::uvec(lmkIdx.n_cols-1);
        size_t k = 0;
        for (size_t j=0; j<lmkIdx.n_cols; ++j) 
		{
            if (lmkIdx.at(0,j) != maxLmk) 
			{
                restCol[k] = j;
                k ++;
            }
        }

        // take the best lmk incrementaly
#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.lock();
#endif
        curMat = curMat - lmkSelectPool[maxLmk].obs_block;
        lmkSelectPool[maxLmk].selected = false;
#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.unlock();
#endif
        //
        lmkIdx = lmkIdx.cols(restCol);
        lmkVisited = lmkVisited.cols(restCol);

    }

    // collect the valid lmk left
    for (size_t i=stIdx; i<edIdx; ++i) 
	{
		if (lmkSelectPool[i].selected == true) 
		{
            //
            subVec.push_back(lmkSelectPool[i]);
        }
    }

    return true;
}

bool Estimator::maxVolSelection_LazierGreedy(const size_t stIdx, const size_t edIdx,
                                                 vector<GoodPoint> &subVec, const size_t mpLimit,
                                                 const double sampleScale) {
	if (lmkSelectPool.size() == 0 )// || subVec == nullptr)
        return false;
    if (stIdx >= edIdx || edIdx > lmkSelectPool.size()) 
        return false;

    if(!subVec.empty())
	    subVec.clear();
  
	if (mpLimit >= edIdx - stIdx) 
	{
        // simply fill in all lmks as selected
        for (size_t i=stIdx; i<edIdx; ++i) 
		{
          #ifdef MULTI_THREAD_LOCK_ON
            mtx_GF.lock();
          #endif
            lmkSelectPool[i].selected = true;
            subVec.push_back(lmkSelectPool[i]);
          #ifdef MULTI_THREAD_LOCK_ON
            mtx_GF.unlock();
          #endif
        }
        //        std::cout << "func maxVolSelection_LazierGreedy: subset limit higher than input lmk number!" << std::endl;
        return true;
    }

    //    std::cout << "Calling lazier greedy!" << std::endl;
    
    // define the size of random subset
    //    double errorBound = 0.001; // 0.005; // 0.01; // 0.0005; //
    size_t szLazierSubset = static_cast<size_t>( double(edIdx - stIdx) / double(mpLimit) * sampleScale );
    //        std::cout << "lazier subset size = " << szLazierSubset << std::endl;
    //    size_t szLazierSubset = static_cast<size_t>( double(mpVec->size()) * 0.3 );
    // for random permutation
    //    std::default_random_engine rndEngine; // or other engine as std::mt19937
    //    std::vector<size_t> rndEntry (szLazierSubset);

    // iteratively search for the most informative lmk
    arma::mat curMat = arma::eye( size(lmkSelectPool[0].obs_block) ) * 0.00001;

    // create a query index for fast insert and reject of entries
    arma::mat lmkIdx = arma::mat(1, edIdx - stIdx);
    // create a flag array to avoid duplicate visit
    arma::mat lmkVisited = arma::mat(1, edIdx - stIdx);
    size_t l = 0;
	//将lmkIdx中的值赋值为路标点的id
    for (size_t i=stIdx; i<edIdx; ++i) 
	{
        lmkIdx.at(0, l) = i;
        lmkVisited.at(0, l) = -1;
        ++ l;
    }
  
    for (size_t i=0; i<mpLimit; ++i) 
	{
        int maxLmk = -1;
        double maxDet = -DBL_MAX;
        //        std::cout << "func maxVolSelection_LazierGreedy: finding subset idx " << i << std::endl;

        size_t numHit = 0, numRndQue = 0, szActualSubset;//H指的公式9中的H
        szActualSubset = szLazierSubset;
        if (lmkIdx.n_cols < szActualSubset)
            szActualSubset = lmkIdx.n_cols;
        //
        //        std::srand(std::time(nullptr));
        while (numHit < szActualSubset) 
		{
            // generate random query index
            //            cout << "random query round " << numHit << endl;
            size_t j;
            numRndQue = 0;
            while (numRndQue < MAX_RANDOM_QUERY_TIME) 
			{
                j = ( std::rand() % lmkIdx.n_cols );
                //                cout << j << " ";
                // check if visited
                if (lmkVisited.at(0, j) < i) {
                    lmkVisited.at(0, j) = i;
                    break ;
                }
                ++ numRndQue;
            }
            if (numRndQue >= MAX_RANDOM_QUERY_TIME)
                break ;

            //            cout << endl;
            int queIdx = lmkIdx.at(0, j);
            ++ numHit;

            if (lmkSelectPool[queIdx].selected == true) 
			{
                //                std::cout << "It does not supposed to happen: " << lmkSelectPool[queIdx].selected << "; "
                //                          <<  lmkSelectPool[queIdx].obs_score << std::endl;
                -- numHit;
                continue ;
            }
            
            arma::cx_double curDet = arma::log_det( curMat + lmkSelectPool[queIdx].obs_block );
            //#endif
            if (curDet.real() > maxDet) 
			{
                maxDet = curDet.real();
                maxLmk = queIdx;
                //                std::cout << "current determinant = " << curDet << std::endl;
            }
        }

        if (maxLmk == -1) 
		{
            std::cout << "func maxVolSelection_LazierGreedy: early termination!" << std::endl;
            break ;
        }

        // set up the index for columns that are not selected yet
        arma::uvec restCol = arma::uvec(lmkIdx.n_cols-1);
        size_t k = 0;
        for (size_t j=0; j<lmkIdx.n_cols; ++j) 
		{
            if (lmkIdx.at(0,j) != maxLmk) 
			{
                restCol[k] = j;
                k ++;
            }
        }

        // take the best lmk incrementaly
#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.lock();
#endif
        curMat = curMat + lmkSelectPool[maxLmk].obs_block;
        lmkSelectPool[maxLmk].selected = true;
        subVec.push_back(lmkSelectPool[maxLmk]);
#ifdef MULTI_THREAD_LOCK_ON
        mtx_GF.unlock();
#endif
        //
        lmkIdx = lmkIdx.cols(restCol);
        lmkVisited = lmkVisited.cols(restCol);
    }

    return true;
}





