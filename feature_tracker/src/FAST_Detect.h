#include "fast.h"
#include "vision.h"

/*
struct Corner
{
  int x;        //!< x-coordinate of corner in the image.
  int y;        //!< y-coordinate of corner in the image.
  float score;  //!< shi-tomasi score of the corner.
  Corner(int x, int y, float score) :
    x(x), y(y), score(score)
  {}
};
typedef vector<Corner> Corners;
*/

void FAST_Detect(const cv::Mat img, cv::Mat Out);
