#include "goodFeaturesToTrack.h"
#include "precomp.hpp"
#include "FAST_Detect.h"
#include <vector>
#include <algorithm>
#include <ros/console.h>

using namespace cv;

void goodFeaturesToTrack( cv::Mat& _image, vector<Point2f>& _corners,
                              int maxCorners, double qualityLevel, double minDistance,
                              cv::Mat& _mask, int blockSize,
                              bool useHarrisDetector, double harrisK)
{
    CV_Assert( qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0 );
    CV_Assert( _mask.empty() || (_mask.type() == CV_8UC1 && _mask.size() == _image.size()) );

    Mat eig = Mat::zeros(_image.rows, _image.cols, CV_32FC1); 
    Mat tmp;
    if( useHarrisDetector )
        cornerHarris( _image, eig, blockSize, 3, harrisK );
    else
    {
      FAST_Detect(_image, eig);//cornerMinEigenVal( image, eig, blockSize, 3 );
	}
        

    double maxVal = 0;
    minMaxLoc( eig, 0, &maxVal, 0, 0, _mask );
    threshold( eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO );
    dilate( eig, tmp, Mat());

    Size imgsize = _image.size();

    vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = _mask.data ? _mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::sort( tmpCorners.begin(), tmpCorners.end());
    vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if(minDistance >= 1)
    {
         // Partition the image into larger grids
        int w = _image.cols;
        int h = _image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    vector <Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                // printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
                //    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(Point2f((float)x, (float)y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    for (auto &it : corners)
    {
      _corners.push_back(it);
    }
	
    //Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);

    /*
    for( i = 0; i < total; i++ )
    {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
        int y = (int)(ofs / eig.step);
        int x = (int)((ofs - y*eig.step)/sizeof(float));

        if( minDistance > 0 )
        {
            for( j = 0; j < ncorners; j++ )
            {
                float dx = x - corners[j].x;
                float dy = y - corners[j].y;
                if( dx*dx + dy*dy < minDistance )
                    break;
            }
            if( j < ncorners )
                continue;
        }

        corners.push_back(Point2f((float)x, (float)y));
        ++ncorners;
        if( maxCorners > 0 && (int)ncorners == maxCorners )
            break;
    }
*/
}

/*
CV_IMPL void
cvGoodFeaturesToTrack( const void* _image, void*, void*,
                       CvPoint2D32f* _corners, int *_corner_count,
                       double quality_level, double min_distance,
                       const void* _maskImage, int block_size,
                       int use_harris, double harris_k )
{
    cv::Mat image = cv::cvarrToMat(_image), mask;
    cv::vector<cv::Point2f> corners;

    if( _maskImage )
        mask = cv::cvarrToMat(_maskImage);

    CV_Assert( _corners && _corner_count );
    cv::goodFeaturesToTrack( image, corners, *_corner_count, quality_level,
        min_distance, mask, block_size, use_harris != 0, harris_k );

    size_t i, ncorners = corners.size();
    for( i = 0; i < ncorners; i++ )
        _corners[i] = corners[i];
    *_corner_count = (int)ncorners;
}
*/
