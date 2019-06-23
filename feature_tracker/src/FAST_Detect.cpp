/**
  @此内容来自SVO
 */

//#include <svo/feature_detection.h>
//#include <svo/feature.h>
#include "FAST_Detect.h"
#include <ros/console.h>

using namespace std;

void FAST_Detect(const cv::Mat img, cv::Mat Out)
{
    vector<fast::fast_xy> fast_corners;
#if __SSE2__
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img.data, img.cols,
          img.rows, img.cols, 20, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img.data, img.cols,
          img.rows, img.cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img.data, img.cols,
          img.rows, img.cols, 20, fast_corners);
#endif
    vector<int> scores, nm_corners;
    fast::fast_corner_score_10((fast::fast_byte*) img.data, img.cols, fast_corners, 20, scores);
	fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      
      const float score = vk::shiTomasiScore(img, xy.x, xy.y);
      Out.at<float>(xy.y, xy.x) = score;
	  
    }

}
