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

StateEstimator::StateEstimator(int _max_iteration, int _min_correspondence_threshold, double _huber_loss_delta,
							   double _converge_threshold, double _lidar_noise_std, double gyro_noise_std,
							   double gyro_bias_noise_std, double acc_noise_std, double acc_bias_noise_std,
							   double gravity_noise_std)
	: current_state(), g_initialized(false) {
	max_iteration = _max_iteration;
	min_correspondence_threshold = _min_correspondence_threshold;
	huber_loss_delta = _huber_loss_delta;
	converge_threshold = _converge_threshold;
	lidar_noise_std = _lidar_noise_std;
	process_noise = Eigen::Matrix<double, 18, 18>::Identity();
	process_noise.block<3, 3>(0, 0) *= gyro_noise_std * gyro_noise_std;
	process_noise.block<3, 3>(3, 3) *= acc_noise_std * acc_noise_std;
	process_noise.block<3, 3>(6, 6) *= acc_noise_std * acc_noise_std;
	process_noise.block<3, 3>(9, 9) *= gyro_bias_noise_std * gyro_bias_noise_std;
	process_noise.block<3, 3>(12, 12) *= acc_bias_noise_std * acc_bias_noise_std;
	process_noise.block<3, 3>(15, 15) *= gravity_noise_std * gravity_noise_std;
}

StateEstimator::~StateEstimator() {}

bool StateEstimator::getGravityInit(const std::vector<IMUData> &imu_buffer) {
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	if (g_initialized)
		return false;

	Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();
	Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();

	for (const auto &imu : imu_buffer) {
		mean_gyro += imu.gyro;
		mean_acc += imu.acc;
	}
	mean_gyro /= imu_buffer.size();
	mean_acc /= imu_buffer.size();
	double var_gyro = 0.0, var_acc = 0.0;

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

	Eigen::Vector3d up = Eigen::Vector3d(0, 0, 9.81);
	Eigen::Vector3d measured_gravity = mean_acc.normalized() * 9.81;
	Eigen::Matrix3d R_align = Eigen::Quaterniond::FromTwoVectors(measured_gravity, up).toRotationMatrix();
	current_state.gravity = -measured_gravity;
	current_state.rotation = Sophus::SO3d(R_align) * current_state.rotation;
	current_state.position = R_align * current_state.position;
	current_state.velocity = R_align * current_state.velocity;
	current_state.gravity = R_align * current_state.gravity;

	if (var_gyro < 1e-5)
		current_state.bias_gyro = mean_gyro;
	else
		current_state.bias_gyro.setZero();

	current_state.bias_acc = mean_acc - R_align.transpose() * up;
	g_initialized = true;
	current_state.covariance.block<3, 3>(0, 0) *= 0.0001f;
	current_state.covariance.block<3, 3>(3, 3) *= 0.001f;
	current_state.covariance.block<3, 3>(6, 6) *= 1.0f;
	current_state.covariance.block<3, 3>(9, 9) *= 0.0001f;
	current_state.covariance.block<3, 3>(12, 12) *= 0.001f;
	current_state.covariance.block<3, 3>(15, 15) *= 0.0001f;

	return true;
}

void StateEstimator::propagateIMU(const IMUData &imu_data) {
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	Eigen::Vector3d omega = imu_data.gyro - current_state.bias_gyro;
	Eigen::Vector3d acc = imu_data.acc - current_state.bias_acc;
	Sophus::SO3d R = current_state.rotation;
	double dt = imu_data.dt;

	Sophus::SO3d dR = Sophus::SO3d::exp(omega * dt);
	Sophus::SO3d R_new = current_state.rotation * dR;
	Eigen::Vector3d acc_world = current_state.rotation * acc + current_state.gravity; // acceleration in world frame
	Eigen::Vector3d new_position = current_state.position + current_state.velocity * dt + 0.5 * acc_world * dt * dt;
	Eigen::Vector3d new_velocity = current_state.velocity + acc_world * dt;

	Eigen::Matrix<double, 18, 18> state_transition = Eigen::Matrix<double, 18, 18>::Identity();
	// Rotation Dynamics
	Eigen::Matrix3d omega_skew = Sophus::SO3d::hat(omega);
	state_transition.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - omega_skew * dt;
	state_transition.block<3, 3>(0, 9) =
		-Sophus::SO3d::leftJacobian(-omega * dt) * dt; //  −J_r(ω dt)·dt = −J_l(−ω dt)·dt

	// Position Dynamics
	Eigen::Matrix3d acc_skew = Sophus::SO3d::hat(acc);
	state_transition.block<3, 3>(3, 0) = -0.5f * R.matrix() * acc_skew * dt * dt;
	state_transition.block<3, 3>(3, 3) = Sophus::Matrix3d::Identity();
	state_transition.block<3, 3>(3, 6) = Sophus::Matrix3d::Identity() * dt;
	state_transition.block<3, 3>(3, 12) = -0.5f * R.matrix() * dt * dt;
	state_transition.block<3, 3>(3, 15) = 0.5f * Sophus::Matrix3d::Identity() * dt * dt;

	// Velocity Dynamics
	state_transition.block<3, 3>(6, 0) = -R.matrix() * acc_skew * dt;
	state_transition.block<3, 3>(6, 6) = Sophus::Matrix3d::Identity();
	state_transition.block<3, 3>(6, 12) = -R.matrix() * dt;
	state_transition.block<3, 3>(6, 15) = Sophus::Matrix3d::Identity() * dt;

	state_transition.block<3, 3>(9, 9) = Sophus::Matrix3d::Identity();
	state_transition.block<3, 3>(12, 12) = Sophus::Matrix3d::Identity();
	state_transition.block<3, 3>(15, 15) = Sophus::Matrix3d::Identity();

	Eigen::Matrix<double, 18, 18> P = current_state.covariance;
	current_state.covariance = state_transition * P * state_transition.transpose() + process_noise * dt;

	current_state.rotation = R_new;
	current_state.position = new_position;
	current_state.velocity = new_velocity;
	StateWithStamp state_stamp = StateWithStamp(imu_data.timestamp, omega, acc, current_state.rotation,
												current_state.position, current_state.velocity, current_state.gravity);
	preint_list.push_back(state_stamp);
}

void StateEstimator::undistortPointcloud(std::vector<PointCloud> &points) {
	std::deque<StateWithStamp> tmp_preint_list;
	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		tmp_preint_list = preint_list;
	}

	if (points.empty() || tmp_preint_list.empty()) {
		spdlog::warn("[Estimator] skipping undistortion: {} points, {} preintegrated states", points.size(),
					 tmp_preint_list.size());
		return;
	}

	auto end_it = upper_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points.back().timestamp,
							  [](double value, const StateWithStamp &a) { return value < a.timestamp; });
	if (end_it != tmp_preint_list.begin())
		end_it--;

	Sophus::SE3d T_imulastpoint_odom = Sophus::SE3d(end_it->rotation, end_it->position).inverse();

	for (size_t iter = 0; iter < points.size(); iter++) {
		auto it = upper_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points[iter].timestamp,
							  [](double value, const StateWithStamp &a) { return value < a.timestamp; });
		if (it != tmp_preint_list.begin())
			it--;
		double dt = points[iter].timestamp - it->timestamp;
		Sophus::SO3d dR = Sophus::SO3d::exp(it->gyro * dt);
		Sophus::SO3d R = it->rotation * dR;

		Eigen::Vector3d acc_world = R * it->accel + it->gravity;
		Eigen::Vector3d pos = it->position + it->velocity * dt + 0.5 * acc_world * dt * dt;
		Sophus::SE3d T_odom_imu(R, pos);
		points[iter].xyz = T_imulastpoint_odom * T_odom_imu * points[iter].xyz;
	}

	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		while (!preint_list.empty() && preint_list.front().timestamp < points.front().timestamp)
			preint_list.pop_front();
	}

	tmp_preint_list.clear();
}

State StateEstimator::updateState(std::vector<Eigen::Vector3d> &points,
								  const std::shared_ptr<VoxelLocalMap> &voxel_local_map) {
	auto start = std::chrono::steady_clock::now();
	State tmp_state;
	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		tmp_state = current_state;
	}

	State prior = tmp_state;
	Eigen::Matrix<double, 18, 18> I18 = Eigen::Matrix<double, 18, 18>::Identity();
	Eigen::Matrix<double, 18, 18> G = Eigen::Matrix<double, 18, 18>::Zero();
	Eigen::Matrix<double, 18, 18> HT_R_inv_H = Eigen::Matrix<double, 18, 18>::Zero();
	Eigen::Matrix<double, 18, 18> K = Eigen::Matrix<double, 18, 18>::Zero();

	for (int iter = 0; iter < max_iteration; iter++) {
		std::vector<Eigen::Vector3d> points_world;
		for (auto it : points)
			points_world.push_back(tmp_state.rotation * it + tmp_state.position);
		auto correspondence = voxel_local_map->findCorrespondence(points_world, 1.0);
		int correspondence_num = static_cast<int>(correspondence.size());

		if (correspondence_num < min_correspondence_threshold) {
			spdlog::info("Insufficient correspondence {}, skipping update.", correspondence_num);
			if (iter == 0) {
				for (auto &p : points)
					p = tmp_state.rotation * p + tmp_state.position;
				return tmp_state;
			}
			break;
		}

		Eigen::MatrixXd H(correspondence_num, 18);
		Eigen::VectorXd r(correspondence_num);
		Eigen::VectorXd R_inv(correspondence_num);
		std::vector<double> residual_vec(correspondence_num), huber_weights(correspondence_num);

		for (int i = 0; i < correspondence_num; i++) {
			auto &point = points[correspondence[i].point_idx];
			auto &normal = correspondence[i].normal;
			auto &centroid = correspondence[i].centroid;
			// Computing Jacobian
			Eigen::Matrix3d point_skew = Sophus::SO3d::hat(point);
			H.row(i).setZero();
			H.row(i).block<1, 3>(0, 0) = -normal.transpose() * tmp_state.rotation.matrix() * point_skew;
			H.row(i).block<1, 3>(0, 3) = normal.transpose();
			const double residual = normal.dot(points_world[correspondence[i].point_idx] - centroid);
			residual_vec[i] = residual;
			r(i) = residual;
		}

		// Robust centre and spread of the residuals: median and MAD.
		// Outliers can't stretch these the way they stretch mean/std.
		std::vector<double> tmp(residual_vec);
		auto mid = tmp.begin() + tmp.size() / 2;
		std::nth_element(tmp.begin(), mid, tmp.end());
		const double med = *mid;
		for (auto &v : tmp)
			v = std::abs(v - med);
		std::nth_element(tmp.begin(), mid, tmp.end());
		const double sigma = std::max(1.4826 * (*mid), 1e-6);

		// Huber weights, measured from the centre so a shared pose offset
		// (the signal) is not mistaken for outliers.
		for (int i = 0; i < correspondence_num; i++) {
			const double residual_abs = std::abs(residual_vec[i] - med) / sigma;
			huber_weights[i] = (residual_abs > huber_loss_delta) ? huber_loss_delta / residual_abs : 1.0;
			R_inv(i) = huber_weights[i] / (lidar_noise_std * lidar_noise_std);
		}

		Eigen::MatrixXd H_9 = H.block(0, 0, correspondence_num, 9);
		Eigen::MatrixXd HT_R_inv(9, correspondence_num);

		for (int i = 0; i < correspondence_num; i++) {
			HT_R_inv.col(i) = H_9.row(i).transpose() * R_inv(i);
		}

		Eigen::Matrix<double, 9, 9> HT_R_inv_H_9 = HT_R_inv * H_9;
		Eigen::Matrix<double, 9, 1> HT_R_inv_r = HT_R_inv * r;
		Eigen::Matrix<double, 18, 18> P_prior = tmp_state.covariance;
		HT_R_inv_H.block<9, 9>(0, 0) = HT_R_inv_H_9;
		Eigen::Matrix<double, 18, 18> information = HT_R_inv_H + P_prior.inverse();
		K = information.inverse();
		Eigen::Matrix<double, 18, 1> dx_prior;
		dx_prior.segment<3>(0) = (prior.rotation.inverse() * tmp_state.rotation).log();
		dx_prior.segment<3>(3) = tmp_state.position - prior.position;
		dx_prior.segment<3>(6) = tmp_state.velocity - prior.velocity;
		dx_prior.segment<3>(9) = tmp_state.bias_gyro - prior.bias_gyro;
		dx_prior.segment<3>(12) = tmp_state.bias_acc - prior.bias_acc;
		dx_prior.segment<3>(15) = tmp_state.gravity - prior.gravity;
		G.block<18, 9>(0, 0) = K.block<18, 9>(0, 0) * HT_R_inv_H_9;
		Eigen::VectorXd dx = -K.block<18, 9>(0, 0) * HT_R_inv_r - (I18 - G) * dx_prior;

		// State correction
		tmp_state.rotation = tmp_state.rotation * Sophus::SO3d::exp(dx.segment<3>(0));
		tmp_state.position += dx.segment<3>(3);
		tmp_state.velocity += dx.segment<3>(6);
		tmp_state.bias_gyro += dx.segment<3>(9);
		tmp_state.bias_acc += dx.segment<3>(12);
		tmp_state.gravity += dx.segment<3>(15);

		spdlog::info("Number of Correspondence {}, IEKF update iter {}, dx norm: {}", correspondence_num, iter,
					 dx.norm());
		if (dx.norm() < converge_threshold)
			break;
	}

	// Joseph form for numerical stability
	Eigen::Matrix<double, 18, 18> I_KH = I18 - G;
	Eigen::Matrix<double, 18, 18> KRKt = K * HT_R_inv_H * K;
	tmp_state.covariance = I_KH * prior.covariance * I_KH.transpose() + KRKt;

	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		current_state = tmp_state;
	}

	for (auto &p : points)
		p = tmp_state.rotation * p + tmp_state.position;

	auto end = std::chrono::steady_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	spdlog::info("[Estimator] updating state took {} ms", elapsed_ms);

	return tmp_state;
}

} // namespace slio