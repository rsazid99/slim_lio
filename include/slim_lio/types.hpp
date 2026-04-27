/*
 * @file            include/slim_lio/types.hpp
 * @description     
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-26 02:08:53
 * @lastModified    2026-04-26 02:10:01
 * Copyright ©Sazid Rahman Simanto All rights reserved
*/

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <math.h>
#include <sophus/se3.hpp>
#include <vector>

namespace slio {

struct IMUData {
    double dt;
    Eigen::Vector3f gyro, acc;

    IMUData(double _dt, const Eigen::Vector3f _gyro, const Eigen::Vector3f _acc) {
        dt = _dt;
        gyro = _gyro;
        acc = _acc;
    }
};

struct PointCloud {
    float x, y, z, intensity, timestamp;

    PointCloud(float _x, float _y, float _z, float _intensity, float _timestamp) {
        x = _x, y = _y, z = _z, intensity = _intensity, timestamp = _timestamp;
    }
};

struct State {
    Eigen::Vector3f position, velocity, bias_gyro, bias_acc, gravity;
    Eigen::Matrix3f rotation;

    Eigen::Matrix<float, 18, 18> covariance;

    State() {
        rotation = Eigen::Matrix3f::Identity();
        position = velocity = bias_gyro = bias_acc = Eigen::Vector3f::Zero();
        gravity = Eigen::Vector3f(0, 0, -9.81);
        covariance = Eigen::Matrix<float, 18, 18>::Identity() * 0.001;
    }
};

} // namespace slio

#endif // TYPES_H