#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();//实际上包含了特征点速度的计算和特征点矫正

    cv::Mat mask;
    cv::Mat fisheye_mask;
	// prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;//光流法跟踪上的特征点个数不足最小特征点数量要求时，额外提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//**pts指的是图像中的像素坐标
    vector<cv::Point2f> prev_un_pts, cur_un_pts;//**un_pts则指的是将**pts根据成像关系反投影到相机坐标系中，归一化z轴坐标而得到的坐标为[x,y,1]的相机坐标系点,用于计算相机的运动速度，见undistortedPoints()函数
    vector<cv::Point2f> pts_velocity;//将跟踪上的特征点反投影到z轴为1的相机坐标系中，计算得到的跟踪点的空间运动速度
    vector<int> ids;
    vector<int> track_cnt;//特征点跟踪次数计数器，cnt表示count缩写
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};
