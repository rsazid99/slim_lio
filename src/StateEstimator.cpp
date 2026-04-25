/*
 * @file            src/StateEstimator.cpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-23 15:45:34
 * @lastModified    2026-04-23 15:45:34
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#include "slim_lio/StateEstimator.hpp"

namespace slio {

StateEstimator::StateEstimator()
    : current_state(), g_initialized(false), g_imu_dt(0.0) {
}

StateEstimator::~StateEstimator() {
}

bool StateEstimator::getGravityInit(const std::vector<IMUData> &imu_buffer) {
    if (g_initialized)
        return false;

    Eigen::Vector3f mean_gyro = Eigen::Vector3f::Zero();
    Eigen::Vector3f mean_acc = Eigen::Vector3f::Zero();

    for (const auto &imu : imu_buffer) {
        mean_gyro += imu.gyro;
        mean_acc += imu.acc;
        g_imu_dt += imu.dt;
    }
    mean_gyro /= imu_buffer.size();
    mean_acc /= imu_buffer.size();
    g_imu_dt /= imu_buffer.size();

    float var_gyro = 0.0, var_acc = 0.0;

    for (const auto &imu : imu_buffer) {
        var_gyro += (imu.gyro - mean_gyro).squaredNorm();
        var_acc += (imu.acc - mean_acc).squaredNorm();
    }

    var_gyro /= imu_buffer.size();
    var_acc /= imu_buffer.size();

    if (var_acc >= 0.5) {
        spdlog::info("[StateEstimator] High accelerometer variance ({:.3f}), robot maybe moving.", var_acc);
    }

    if (var_gyro >= 0.5) {
        spdlog::info("[StateEstimator] High gyroscope variance ({:.3f}), robot maybe rotating.", var_gyro);
    }

    Eigen::Vector3f up = Eigen::Vector3f(0, 0, 9.81);
    Eigen::Vector3f measured_gravity = mean_acc.normalized() * 9.81;
    Eigen::Matrix3f R_align = Eigen::Quaternionf::FromTwoVectors(measured_gravity, up).toRotationMatrix();
    current_state.gravity = -measured_gravity;

    current_state.rotation = R_align * current_state.rotation;
    current_state.position = R_align * current_state.position;
    current_state.velocity = R_align * current_state.velocity;
    current_state.gravity = R_align * current_state.gravity;
    current_state.bias_gyro = mean_gyro;
    current_state.bias_acc = mean_acc - R_align.transpose() * up;
    spdlog::info("[StateEstimator] Calculated gravity [{:3f}, {:3f}, {:3f}], accelerometer bias [{:3f}, {:3f}, {:3f}]",
                 current_state.gravity.x(), current_state.gravity.y(), current_state.gravity.z(), current_state.bias_acc.x(),
                 current_state.bias_acc.y(), current_state.bias_acc.z());
    g_initialized = true;

    return true;
}

} // namespace slio