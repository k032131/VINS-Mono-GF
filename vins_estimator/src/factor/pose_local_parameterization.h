#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };//位置3维，姿态四元数4维
    virtual int LocalSize() const { return 6; };//位置3维，扰动角度theta，四元数中第一项值始终为1所以只有三维
};
