/*
 * @file            include/slim_lio/StateEstimator.hpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-23 13:48:48
 * @lastModified    2026-04-23 16:29:53
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#ifndef STATEESTIMATOR_H
#define STATEESTIMATOR_H

#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <sophus/se3.hpp>
#include <iostream>
#include "spdlog/spdlog.h"

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

class StateEstimator {
  public:
    StateEstimator();
    ~StateEstimator();

    bool getGravityInit(const std::vector<IMUData> &imu_buffer);

  private:
    State current_state;
    bool g_initialized;
    double g_imu_dt;
};

} // namespace slio

#endif // STATEESTIMATOR_H