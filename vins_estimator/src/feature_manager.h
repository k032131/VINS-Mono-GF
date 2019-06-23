#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "armadillo"

////每一个路标点在一张图像中的信息
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
		
    }
    double cur_td;//时间偏差（摄像头和IMU之间的时间差）
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;

	//used for good features selection
	arma::mat ObsMat;
	
};

//以feature_id为索引，并保存了出现该路标点的第一帧id
class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;//特征首次观测到的特征帧
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;////滑窗中某一个路标点被跟踪到了多少次
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    //used for good features selection
	bool selected;//flag of the good feature selection


    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
       selected = false;
    }

    int endFrame();
};

/**
 * @brief 用来获取滑窗内所有的路标点
 *
 * FeatureManger
 * * *通过list<FeaturePerId> feature 来获得,featurePerId意义：每个路标点由多个连续的图像观测到
 * * * * *由FeaturePerFrame来记录每个路标点在一张图像中的信息，包括三维位置，图像位置，速度等等
 *总结：FeatureManger主要用来获得窗口内所有的路标点;而每一个路标点可以被很多帧观测到，该信息被存放在FeaturePerId中；具体每一个路标点在每一帧观测到的图像中的信息存放在FeaturePerFrame中。这三个类是层层递进的关系，逐步实现了各个信息的存储
 */
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;//得到滑动窗口内所有路标点信息
    list<FeaturePerId> feature_selected;//used by the good feature selection
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif