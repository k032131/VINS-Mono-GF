/**
  @brief 此内容来自opencv
 */

#include "precomp.hpp"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>


namespace cv
{

template<typename T> struct greaterThanPtr
{
    bool operator()(const T* a, const T* b) const { return *a > *b; }
};

}

void goodFeaturesToTrack( cv::Mat& _image, vector<cv::Point2f>& _corners,
                              int maxCorners, double qualityLevel, double minDistance,
                              cv::Mat& _mask, int blockSize = 3,
                              bool useHarrisDetector = false, double harrisK = 0.04);

/*
CV_IMPL void
cvGoodFeaturesToTrack( const void* _image, void*, void*,
                       CvPoint2D32f* _corners, int *_corner_count,
                       double quality_level, double min_distance,
                       const void* _maskImage, int block_size = 3,
                       int use_harris = 0, double harris_k = 0.04);
*/
