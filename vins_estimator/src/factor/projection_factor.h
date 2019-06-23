#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

/**
 * @brief ceres中使用的残差项
 * 
 * 创建一个类继承ceres::CostFunction
 * 重写Evaluate()函数来定义residual的计算形式
 * 对于观测数大于2的路标点，首次观测与后面每一次观测之间各一个residual项
 */
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>//7:pose1;7:pose2;7:extrinsic;1:inv_depth;
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
