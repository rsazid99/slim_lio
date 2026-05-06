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
    : current_state(), g_initialized(false), max_iteration(5), min_correspondence_threshold(50) {
}

StateEstimator::~StateEstimator() {
}

bool StateEstimator::getGravityInit(const std::vector<IMUData>& imu_buffer) {
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
    // spdlog::info("[StateEstimator] Calculated gravity [{:3f}, {:3f}, {:3f}], accelerometer bias [{:3f}, {:3f}, {:3f}]",
    //              current_state.gravity.x(), current_state.gravity.y(), current_state.gravity.z(), current_state.bias_acc.x(),
    //              current_state.bias_acc.y(), current_state.bias_acc.z());

    g_initialized = true;

    current_state.covariance.block<3, 3>(0, 0) *= 0.01f;
    current_state.covariance.block<3, 3>(3, 3) *= 1.0f;
    current_state.covariance.block<3, 3>(6, 6) *= 0.1f;
    current_state.covariance.block<3, 3>(9, 9) *= 0.0001f;
    current_state.covariance.block<3, 3>(12, 12) *= 0.001f;
    current_state.covariance.block<3, 3>(15, 15) *= 0.001f;

    return true;
}

void StateEstimator::propagateIMU(const IMUData& imu_data) {
    Eigen::Vector3f omega = imu_data.gyro - current_state.bias_gyro;
    Eigen::Vector3f acc = imu_data.acc - current_state.bias_acc;
    double dt = imu_data.dt;

    Sophus::SO3f dR = Sophus::SO3f::exp(omega * dt);
    Sophus::SO3f R = current_state.rotation * dR;
    Eigen::Vector3f acc_world = current_state.rotation * acc + current_state.gravity; // acceleration in world frame

    StateWithStamp state_stamp = StateWithStamp(imu_data.timestamp, omega, acc,
                                                current_state.rotation, current_state.position, current_state.velocity, current_state.gravity);
    {
        std::lock_guard<std::mutex> lock(state_estimator_mutex);
        preint_list.push_back(state_stamp);
    }

    current_state.rotation = R;
    current_state.position += current_state.velocity * dt + 0.5 * acc_world * dt * dt;
    current_state.velocity += acc_world * dt;

    // To do -- update covariance matrix ---
    //spdlog::info("propagated imu measurement time: {}", imu_data.timestamp);

}

void StateEstimator::undistortPointcloud(std::vector<PointCloud>& points) {
    std::deque<StateWithStamp> tmp_preint_list;
    {
        std::lock_guard<std::mutex> lock(state_estimator_mutex);
        tmp_preint_list = preint_list;
    }
    auto end_it = upper_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points.back().timestamp, [](double value, const StateWithStamp &a) {
        return value < a.timestamp;
    });
    //spdlog::info("The upperbound tstamp {}, last pointcloud tstamp {}", end_it->timestamp, points.back().timestamp);
    Sophus::SE3f T_imulastpoint_odom = Sophus::SE3f(end_it->rotation, end_it->position).inverse();
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
        points[iter].xyz = T_imulastpoint_odom * T_odom_imu * points[iter].xyz;
    }
    {
        std::lock_guard<std::mutex> lock(state_estimator_mutex);
        while(!preint_list.empty() && preint_list.front().timestamp < points.front().timestamp) preint_list.pop_front();
    }
    tmp_preint_list.clear();
    //spdlog::info("The size of preint queue is {}", preint_list.size());
}

Eigen::Matrix<double, 1, 18> StateEstimator::computeJacobian(const Eigen::Vector3f& point, const Eigen::Vector3f normal) {
    Eigen::Matrix<double, 1, 18> H = Eigen::Matrix<double, 1, 18>::Zero();
    Eigen::Matrix3d point_skew = Sophus::SO3d::hat(point);
    H.block<1, 3>(0, 0) = -normal.transpose() * current_state.rotation.matrix() * point_skew;
    H.block<1, 3>(0, 3) = normal.transpose();

    return H;
}

void StateEstimator::updateState(std::vector<Eigen::Vector3f>& points, const std::shared_ptr<VoxelLocalMap>& voxel_local_map) {
     
    for(size_t iter = 0; iter < max_iteration; iter++) {
        std::vector<Eigen::Vector3f> points_world;
        for(auto it: points) points_world.push_back(current_state.rotation * it + current_state.position);
        
        auto correspondence  = voxel_local_map->findCorrespondence(points_world, 1.0);
        int correspondence_num = std::count_if(correspondence.begin(), correspondence.end(), [](const auto& x) { return x.found});
        if(correspondence_num < min_correspondence_threshold) {
            spdlog::info("Insufficient correspondence {}, skipping update.", correspondence_num);
            return;
        }

        Eigen::MatrixXf H(correspondence_num, 18);
        Eigen::VectorXf r(correspondence_num);
        Eigen::MatrixXf R_meas = Eigen::MatrixXd::Identity(correspondence_num, correspondence_num) * 0.001;
        int row = 0;
        for(size_t i = 0; i < correspondence.size(); i ++) {
            if(!correspondence[i].found) continue;
            auto& point = points[i];
            auto& normal = correspondence[i].normal;
            auto& centroid = correspondence[i].centroid;

            H.row(row) = computeJacobian(point, normal);
            r(row) = normal.transpose() * (point - centroid);
            row++;
        }

        // Kalman gain
        Eigen::MatrixXd  S = (H * current_state.covariance * H.transpose() + R_meas);
        Eigen::MatrixXd  K = current_state.covariance * H.transpose() * S.inverse();

        // State correction
        Eigen::VectorXd dx = K * r;
        current_state.rotation = current_state.rotation * Sophus::SO3f(dx.segment<3>(0));

    }
}

int StateEstimator::getPreintegrationListSize() {
    {
        std::lock_guard<std::mutex> lock(state_estimator_mutex);
        return preint_list.size();
    }
}

Sophus::SE3f StateEstimator::getCurrentPose() {
    {
        std::lock_guard<std::mutex> lock(state_estimator_mutex);
        return Sophus::SE3f(current_state.rotation, current_state.position);
    }
}

} // namespace slio