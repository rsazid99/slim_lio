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
    : current_state(), g_initialized(false) {
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
    }
    mean_gyro /= imu_buffer.size();
    mean_acc /= imu_buffer.size();

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

    current_state.rotation =  current_state.rotation * Sophus::SO3f(R_align);
    current_state.position = R_align * current_state.position;
    current_state.velocity = R_align * current_state.velocity;
    current_state.gravity = R_align * current_state.gravity;
    current_state.bias_gyro = mean_gyro;
    current_state.bias_acc = mean_acc - R_align.transpose() * up;
    spdlog::info("[StateEstimator] Calculated gravity [{:3f}, {:3f}, {:3f}], accelerometer bias [{:3f}, {:3f}, {:3f}]",
                 current_state.gravity.x(), current_state.gravity.y(), current_state.gravity.z(), current_state.bias_acc.x(),
                 current_state.bias_acc.y(), current_state.bias_acc.z());

    g_initialized = true;

    current_state.covariance.block<3, 3>(0, 0) *= 0.01f;
    current_state.covariance.block<3, 3>(3, 3) *= 1.0f;
    current_state.covariance.block<3, 3>(6, 6) *= 0.1f;
    current_state.covariance.block<3, 3>(9, 9) *= 0.0001f;
    current_state.covariance.block<3, 3>(12, 12) *= 0.001f;
    current_state.covariance.block<3, 3>(15, 15) *= 0.001f;

    return true;
}

void StateEstimator::propagateIMU(const IMUData &imu_data) {
    Eigen::Vector3f omega = imu_data.gyro - current_state.bias_gyro;
    Eigen::Vector3f acc = imu_data.acc - current_state.bias_acc;
    double dt = imu_data.dt;

    Sophus::SO3f dR = Sophus::SO3f::exp(omega * dt);
    Sophus::SO3f R = current_state.rotation * dR;
    Eigen::Vector3f acc_world = current_state.rotation * acc + current_state.gravity; // acceleration in world frame

    StateWithStamp state_stamp = StateWithStamp(imu_data.timestamp, omega, acc,
                                                current_state.rotation, current_state.position, current_state.velocity, current_state.gravity);
    {
        std::lock_guard<std::mutex> lock(preint_mutex);
        preint_list.push_back(state_stamp);
    }

    current_state.rotation = R;
    current_state.position += current_state.velocity * dt + 0.5 * acc_world * dt * dt;
    current_state.velocity += acc_world * dt;

    // To do -- update covariance matrix ---
    spdlog::info("propagated imu measurement time: {}", imu_data.timestamp);

}

void StateEstimator::undistortPointcloud(std::vector<PointCloud> &points) {
    std::deque<StateWithStamp> tmp_preint_list;
    {
        std::lock_guard<std::mutex> lock(preint_mutex);
        tmp_preint_list = preint_list;
    }
    auto end_it = upper_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points.back().timestamp, [](double value, const StateWithStamp &a) {
        return value < a.timestamp;
    });
    spdlog::info("The lowerbound tstamp {}, last pointcloud tstamp {}", end_it->timestamp, points.back().timestamp);
    Sophus::SE3f T_imu_odom = Sophus::SE3f(end_it->rotation, end_it->position).inverse();
    for (size_t iter = 0; iter < points.size(); iter++) {
        auto it = lower_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points[iter].timestamp, [](const StateWithStamp &a, double value) {
            return a.timestamp < value;
        });
        it--;
        double dt = points[iter].timestamp - it->timestamp;
        Sophus::SO3f dR = Sophus::SO3f::exp(it->gyro * dt);
        Sophus::SO3f R = it->rotation * dR;
        
        Eigen::Vector3f acc_world = R * it->accel + it->gravity;
        Eigen::Vector3f pos = it->position + it->velocity * dt + 0.5 * acc_world * dt * dt;
        Sophus::SE3f T_odom_imu(R, pos);
        points[iter].xyz = T_imu_odom * T_odom_imu * T_lidar_imu * points[iter].xyz;
    }
    {
        std::lock_guard<std::mutex> lock(preint_mutex);
        while(!preint_list.empty() && preint_list.front().timestamp < points.front().timestamp) preint_list.pop_front();
    }
    tmp_preint_list.clear();
    spdlog::info("The size of preint queue is {}", preint_list.size());
}

int StateEstimator::getPreintegrationListSize() {
    return preint_list.size();
}

} // namespace slio