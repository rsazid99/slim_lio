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
    double dt, timestamp;
    Eigen::Vector3f gyro, acc;

    IMUData(double _dt, double _timestamp, const Eigen::Vector3f _gyro, const Eigen::Vector3f _acc) {
        dt = _dt;
        timestamp = _timestamp;
        gyro = _gyro;
        acc = _acc;
    }
};

struct PointCloud {
    float intensity;
    double timestamp;
    Eigen::Vector3f xyz;

    PointCloud(float _x, float _y, float _z, float _intensity, double _timestamp) {
        xyz = Eigen::Vector3f(_x, _y, _z);
        intensity = _intensity, timestamp = _timestamp;
    }
};

struct State {
    Eigen::Vector3f position, velocity, bias_gyro, bias_acc, gravity;
    Sophus::SO3f rotation;

    Eigen::Matrix<float, 18, 18> covariance;

    State() {
        rotation = Sophus::SO3f();
        position = velocity = bias_gyro = bias_acc = Eigen::Vector3f::Zero();
        gravity = Eigen::Vector3f(0, 0, -9.81);
        covariance = Eigen::Matrix<float, 18, 18>::Identity() * 0.001;
    }
};

struct StateWithStamp {
    Eigen::Vector3f position, velocity, gyro, accel, gravity;
    Sophus::SO3f rotation;
    double timestamp;

    StateWithStamp(double _tstamp, Eigen::Vector3f _gyro, Eigen::Vector3f _accel, 
        Sophus::SO3f _rot, Eigen::Vector3f _pos, Eigen::Vector3f _vel, Eigen::Vector3f _gravity) {
        timestamp = _tstamp;
        gyro = _gyro;
        accel = _accel;
        rotation = _rot;
        position = _pos;
        velocity = _vel;
        gravity = _gravity;
    }
};

struct VoxelData {
    Eigen::Vector3d sum = Eigen::Vector3f::Zero();
    Eigen::Matrix3d pp_T_sum = Eigen::Matrix3f::Zero();
    int count = 0;
    bool valid = false;
    Eigen::Vector3f nomal = Eigen::Vector3f::Zero();
    float planarity = 0.0;
};

struct Correspondence {
    Eigen::Vector3f point, normal;
};

} // namespace slio

#endif // TYPES_H