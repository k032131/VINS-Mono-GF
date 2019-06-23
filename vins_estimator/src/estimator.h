#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include "armadillo"
#include <thread>



#define MULTI_THREAD_LOCK_ON
#define MAX_RANDOM_QUERY_TIME  2000 


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();

	//used for good features selection
	bool setSelction_Number(const size_t num_good_inlier,
                                       const int greedy_mtd,
                                       const double error_bound,
                                       ImageFrame *pFrame, vector<GoodPoint> &mpVec);
	bool runMatrixBuilding(const bool with_multi_thread) ;
	void batchInfoMat_Frame(const size_t start_idx, const size_t end_idx);
	bool maxVolAutomatic_LazierGreedy(const size_t stIdx, const size_t edIdx,
													 vector<GoodPoint> &subVec, const size_t mpLimit,
													 const double sampleScale);
	bool maxVolDeletion_LazierGreedy(const size_t stIdx, const size_t edIdx,
                                                vector<GoodPoint> &subVec, const size_t mpLimit,
                                                const double sampleScale);
	bool maxVolSelection_LazierGreedy(const size_t stIdx, const size_t edIdx,
                                                 vector<GoodPoint> &subVec, const size_t mpLimit,
                                                 const double sampleScale);


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    bool selection_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];//相机到IMU坐标系的变化矩阵 i:IMU;c:Camera
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];//关键帧在世界坐标系下的位姿,见processImage函数倒数第五行
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;//滑动窗口中帧的数量
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;//保存窗口内帧间的预积分值
    IntegrationBase *tmp_pre_integration;//中间变量，将其赋值给processImage函数中imageframe.pre_integration

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;

	//good feature select variables
	std::thread * mThreads;
	ImageFrame *pFrame;
	ImageFrame selectFrame;
	size_t mNumThreads = std::thread::hardware_concurrency(); // 4;
	vector<GoodPoint> lmkSelectPool;
	vector <GoodPoint>  mpVecSelectedBefore;
	vector <GoodPoint>  mpVecSelected;
	arma::colvec v1 = {1.0, -1.0, -1.0, -1.0};
	arma::mat dqbar_by_dq = arma::diagmat(v1);
	
	inline arma::rowvec qconj(arma::rowvec q) 
	{
      q = -1.0 * q;
      q[0] = -q[0];
      return q;
    }
	
    /*注：此处将相机的内参数改为了固定值，没有按照参数传点，后期需要调整*/
	inline bool compute_H_subblock_simplied (const Matrix3d& last_R, const Vector3d last_P,
											const arma::rowvec & yi,
											arma::mat & H13, arma::mat & H47,
											arma::mat & dhu_dhrl,
											float & u, float & v) {

	   //arma::rowvec q_wr = Xv.subvec(3,6);//相机坐标系四元数
	   arma::mat R_wr = {{last_R(0,0),last_R(0,1),last_R(0,2)},
	                           {last_R(1,0),last_R(1,1),last_R(1,2)},
	                            {last_R(2,0),last_R(2,1),last_R(2,2)}};//arma::inv(q2r(q_wr));//四元数转旋转矩阵
       Eigen::Quaterniond q_eigen = Eigen::Quaterniond(last_R);
	   arma::rowvec q_wr = {q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z()};
	   arma::mat R_rw = arma::inv(R_wr);
	   arma::rowvec yi_C = {last_P(0), last_P(1), last_P(2)};  
	   arma::rowvec t_rw = yi - yi_C;//Xv.subvec(0,2);//路标点相对于相机坐标系的平移

	   // dhu_dhrl
	   // lmk @ camera coordinate
	   arma::mat hrl = R_rw * t_rw.t();//世界坐标系到相机坐标系的变换，路标点在相机坐标系中的坐标

	   //路标点在图像点中的坐标
	   if (hrl(2) > 0) 
	   {
	       //此处将相机的内参数改为了固定值，没有按照参数传点，后期需要调整
		   u = float(461.6/*camera.fu*/) * hrl(0)/hrl(2) + float(363.0/*camera.Cx*/);
		   v = float(460.3/*camera.fv*/) * hrl(1)/hrl(2) + float(248.1/*camera.Cy*/);
	   }
	   else 
	   {
           return false;
	       //u = FLT_MAX;
		   //v = FLT_MAX;
	   }

	   if ( fabs(hrl(2)) < 1e-6 ) 
	   {
		   dhu_dhrl  = arma::zeros<arma::mat>(2,3);
	   } 
	   else 
	   {
		   //u,v对hrl(0,0),hrl(1,0),hrl(2,0)的一阶导数
		   /*dhu_dhrl = { {camera.fu/(hrl(2)), 0.0, -hrl(0)*camera.fu/( std::pow(hrl(2), 2.0))},
						{0.0, camera.fv/(hrl(2)), -hrl(1)*camera.fv/( std::pow(hrl(2), 2.0))} };*/
           //此处将相机的内参数改为了固定值，没有按照参数传点，后期需要调整
		   dhu_dhrl = { {461.6/(hrl(2)), 0.0, -hrl(0)*461.6/( std::pow(hrl(2), 2.0))},
						{0.0, 460.3/(hrl(2)), -hrl(1)*460.3/( std::pow(hrl(2), 2.0))} };
	   }

	   arma::rowvec qwr_conj = qconj(q_wr);//q_wr的共轭

	   // H matrix subblock (cols 1~3): H13
	   H13 = -1.0 * (dhu_dhrl *  R_rw);
  
	   // H matrix subblock (cols 4~7): H47
	   H47 = dhu_dhrl * (dRq_times_a_by_dq( qwr_conj ,	t_rw) * dqbar_by_dq);

	   return true;
    }
											
   inline arma::mat dRq_times_a_by_dq(const arma::rowvec & q,
                                   const arma::rowvec & aMat) {

    //    assert(aMat.n_rows == 3 && aMat.n_cols == 1);
//    assert(aMat.n_cols == 3);

    double q0 = q[0];
    double qx = q[1];
    double qy = q[2];
    double qz = q[3];

    arma::mat dR_by_dq0(3,3), dR_by_dqx(3,3), dR_by_dqy(3,3), dR_by_dqz(3,3);
    //    dR_by_dq0 << 2.0*q0 << -2.0*qz << 2.0*qy << arma::endr
    //              << 2.0*qz << 2.0*q0  << -2.0*qx << arma::endr
    //              << -2.0*qy << 2.0*qx << 2.0*q0 << arma::endr;
    dR_by_dq0 = { {2.0*q0, -2.0*qz, 2.0*qy},
                  {2.0*qz, 2.0*q0, -2.0*qx},
                  {-2.0*qy, 2.0*qx, 2.0*q0} };

    //    dR_by_dqx << 2.0*qx << 2.0*qy << 2.0*qz << arma::endr
    //              << 2.0*qy << -2.0*qx << -2.0*q0 << arma::endr
    //              << 2.0*qz << 2.0*q0 << -2.0*qx << arma::endr;
    dR_by_dqx = { {2.0*qx, 2.0*qy, 2.0*qz},
                  {2.0*qy, -2.0*qx, -2.0*q0},
                  {2.0*qz, 2.0*q0, -2.0*qx} };

    //    dR_by_dqy << -2.0*qy << 2.0*qx << 2.0*q0 << arma::endr
    //              << 2.0*qx << 2.0*qy  << 2.0*qz << arma::endr
    //              << -2.0*q0 << 2.0*qz << -2.0*qy << arma::endr;
    dR_by_dqy = { {-2.0*qy, 2.0*qx, 2.0*q0},
                  {2.0*qx, 2.0*qy, 2.0*qz},
                  {-2.0*q0, 2.0*qz, -2.0*qy} };

    //    dR_by_dqz << -2.0*qz << -2.0*q0 << 2.0*qx << arma::endr
    //              << 2.0*q0 << -2.0*qz  << 2.0*qy << arma::endr
    //              << 2.0*qx << 2.0*qy << 2.0*qz << arma::endr;
    dR_by_dqz = { {-2.0*qz, -2.0*q0, 2.0*qx},
                  {2.0*q0, -2.0*qz, 2.0*qy},
                  {2.0*qx, 2.0*qy, 2.0*qz} };

    arma::mat RES = arma::zeros<arma::mat>(3,4);
    RES(arma::span(0,2), arma::span(0,0)) = dR_by_dq0 * aMat.t();
    RES(arma::span(0,2), arma::span(1,1)) = dR_by_dqx * aMat.t();
    RES(arma::span(0,2), arma::span(2,2)) = dR_by_dqy * aMat.t();
    RES(arma::span(0,2), arma::span(3,3)) = dR_by_dqz * aMat.t();

    return RES;
   }

	inline void reWeightInfoMat(const ImageFrame * F, const int & kptIdx,
									const arma::mat & H_meas, const float & res_u, const float & res_v,
									const arma::mat & H_proj, arma::mat & H_rw) {
	
			int measSz = H_meas.n_rows;
			arma::mat Sigma_r(measSz, measSz), W_r(measSz, measSz);
			Sigma_r.eye();
	
			if (F != NULL && kptIdx >= 0 && kptIdx < F->points.size()) 
			{
				float Sigma2 = 1.0;//F->mvLevelSigma2[F->mvKeysUn[kptIdx].octave];
				Sigma_r = Sigma_r * Sigma2;
			}
	
			// cholesky decomp of diagonal-block scaling matrix W
			if (arma::chol(W_r, Sigma_r, "lower") == true) 
			{
				// scale the meas. Jacobian with the scaling block W_r
				H_rw = arma::inv(W_r) * H_meas;
			}
			else 
			{
				// do nothing
				//					  std::cout << "chol failed!" << std::endl;
				//					  std::cout << "oct level =" << kpUn.octave << "; invSigma2 = " << invSigma2 << std::endl;
			}
	
	}

};
									

