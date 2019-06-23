#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
            int N = this->points.size();
            mvbJacobBuilt = vector<bool>(N,false);
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;

		//used for good features selection
		std::vector<bool> mvbJacobBuilt;
};

class GoodPoint {
public:

    GoodPoint(const size_t& idx,  const arma::mat& obs_matrix) {
        this->idx = idx;
        this->obs_block = obs_matrix;
        this->selected = false;
        //this->upper_bound = -DBL_MAX;
    }

    //
    size_t idx;//==feature_id
    arma::vec obs_vector;
    arma::mat obs_block;//greedy算法中用到，非常重要
    // for acceleration
    arma::mat sum_mat;
    double upper_bound;
    //
    float pI[2];//image point==uv
    float pW[3];//world point==point
    bool selected;//YKang

};


bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);
