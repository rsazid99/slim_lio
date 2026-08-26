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

StateEstimator::StateEstimator(float gyro_noise_std, float gyro_bias_noise_std, float acc_noise_std,
							   float acc_bias_noise_std, float gravity_noise_std)
	: current_state(), g_initialized(false), max_iteration(10), min_correspondence_threshold(20), huber_loss_delta(0.5),
	  converge_threshold(0.01), lidar_noise_std(0.05) {
	process_noise = Eigen::Matrix<float, 18, 18>::Identity();
	process_noise.block<3, 3>(0, 0) *= gyro_noise_std * gyro_noise_std;
	process_noise.block<3, 3>(3, 3) *= acc_noise_std * acc_noise_std;
	process_noise.block<3, 3>(6, 6) *= acc_noise_std * acc_noise_std;
	process_noise.block<3, 3>(9, 9) *= gyro_bias_noise_std * gyro_bias_noise_std;
	process_noise.block<3, 3>(12, 12) *= acc_bias_noise_std * acc_bias_noise_std;
	process_noise.block<3, 3>(15, 15) *= gravity_noise_std * gravity_noise_std;
}

StateEstimator::~StateEstimator() {}

bool StateEstimator::getGravityInit(const std::vector<IMUData> &imu_buffer) {
	// auto start = std::chrono::steady_clock::now();
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
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

	current_state.rotation = Sophus::SO3f(R_align) * current_state.rotation;
	current_state.position = R_align * current_state.position;
	current_state.velocity = R_align * current_state.velocity;
	current_state.gravity = R_align * current_state.gravity;
	current_state.bias_gyro = mean_gyro;
	current_state.bias_acc = mean_acc - R_align.transpose() * up;
	// spdlog::info("[StateEstimator] Calculated gravity [{:3f}, {:3f}, {:3f}], accelerometer bias [{:3f}, {:3f},
	// {:3f}]",
	//              current_state.gravity.x(), current_state.gravity.y(), current_state.gravity.z(),
	//              current_state.bias_acc.x(), current_state.bias_acc.y(), current_state.bias_acc.z());

	g_initialized = true;

	current_state.covariance.block<3, 3>(0, 0) *= 0.01f;
	current_state.covariance.block<3, 3>(3, 3) *= 1.0f;
	current_state.covariance.block<3, 3>(6, 6) *= 0.1f;
	current_state.covariance.block<3, 3>(9, 9) *= 0.0001f;
	current_state.covariance.block<3, 3>(12, 12) *= 0.001f;
	current_state.covariance.block<3, 3>(15, 15) *= 0.001f;
	// auto end = std::chrono::steady_clock::now();
	// double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	// spdlog::info("[Estimator] calculating gravity took {} ms", elapsed_ms);
	return true;
}

void StateEstimator::propagateIMU(const IMUData &imu_data) {
	// auto start = std::chrono::steady_clock::now();
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	Eigen::Vector3f omega = imu_data.gyro - current_state.bias_gyro;
	Eigen::Vector3f acc = imu_data.acc - current_state.bias_acc;
	Sophus::SO3f R = current_state.rotation;
	float dt = static_cast<float>(imu_data.dt);

	Sophus::SO3f dR = Sophus::SO3f::exp(omega * dt);
	Sophus::SO3f R_new = current_state.rotation * dR;
	Eigen::Vector3f acc_world = current_state.rotation * acc + current_state.gravity; // acceleration in world frame
	Eigen::Vector3f new_position = current_state.position + current_state.velocity * dt + 0.5 * acc_world * dt * dt;
	Eigen::Vector3f new_velocity = current_state.velocity + acc_world * dt;

	StateWithStamp state_stamp =
		StateWithStamp(imu_data.timestamp, omega, acc, current_state.rotation, current_state.position,
					   current_state.velocity, current_state.gravity, Sophus::SE3f(R_new, new_position));
	preint_list.push_back(state_stamp);

	Eigen::Matrix<float, 18, 18> state_transition = Eigen::Matrix<float, 18, 18>::Identity();
	// Rotation Dynamics
	Eigen::Matrix3f omega_skew = Sophus::SO3f::hat(omega);
	state_transition.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() - omega_skew * dt;
	state_transition.block<3, 3>(0, 9) = -Sophus::SO3f::leftJacobian(omega * dt) * dt; // J_r(phi) = J_l(-phi)

	// Position Dynamics
	Eigen::Matrix3f acc_skew = Sophus::SO3f::hat(acc);
	state_transition.block<3, 3>(3, 0) = -0.5f * R.matrix() * acc_skew * dt * dt;
	state_transition.block<3, 3>(3, 3) = Sophus::Matrix3f::Identity();
	state_transition.block<3, 3>(3, 6) = Sophus::Matrix3f::Identity() * dt;
	state_transition.block<3, 3>(3, 12) = -0.5f * R.matrix() * dt * dt;
	state_transition.block<3, 3>(3, 15) = 0.5f * Sophus::Matrix3f::Identity() * dt * dt;

	// Velocity Dynamics
	state_transition.block<3, 3>(6, 0) = -R.matrix() * acc_skew * dt;
	state_transition.block<3, 3>(6, 6) = Sophus::Matrix3f::Identity();
	state_transition.block<3, 3>(6, 12) = -R.matrix() * dt;
	state_transition.block<3, 3>(6, 15) = Sophus::Matrix3f::Identity() * dt;

	state_transition.block<3, 3>(9, 9) = Sophus::Matrix3f::Identity();
	state_transition.block<3, 3>(12, 12) = Sophus::Matrix3f::Identity();
	state_transition.block<3, 3>(15, 15) = Sophus::Matrix3f::Identity();

	Eigen::Matrix<float, 18, 18> P = current_state.covariance;
	current_state.covariance = state_transition * P * state_transition.transpose() + process_noise * dt;
	// spdlog::info("propagated imu measurement time: {}", imu_data.timestamp);
	current_state.rotation = R_new;
	current_state.position = new_position;
	current_state.velocity = new_velocity;
	// auto end = std::chrono::steady_clock::now();
	// double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	// spdlog::info("[Estimator] propagating imu measurement took {} ms", elapsed_ms);
}

void StateEstimator::undistortPointcloud(std::vector<PointCloud> &points) {
	// auto start = std::chrono::steady_clock::now();
	//  std::lock_guard<std::mutex> lock(state_estimator_mutex);
	//  std::deque<StateWithStamp> tmp_preint_list;
	//  tmp_preint_list = preint_list;
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
	// spdlog::info("The upperbound tstamp {}, last pointcloud tstamp {}, {}, {}, {}", end_it->timestamp,
	// points.back().timestamp, tmp_preint_list.front().timestamp, tmp_preint_list.back().timestamp,
	// tmp_preint_list.size());
	Sophus::SE3f T_imulastpoint_odom = end_it->pred_pose.inverse();
	for (size_t iter = 0; iter < points.size(); iter++) {
		auto it = lower_bound(tmp_preint_list.begin(), tmp_preint_list.end(), points[iter].timestamp,
							  [](const StateWithStamp &a, double value) { return a.timestamp < value; });
		if (it != tmp_preint_list.begin())
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
		while (!preint_list.empty() && preint_list.front().timestamp < points.front().timestamp)
			preint_list.pop_front();
	}
	tmp_preint_list.clear();
	// auto end = std::chrono::steady_clock::now();
	// double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	// spdlog::info("[Estimator] undistorting pointcloud took {} ms", elapsed_ms);
	// spdlog::info("After undistortPointcloud the size of preint queue is {}", preint_list.size());
}

State StateEstimator::updateState(std::vector<Eigen::Vector3f> &points,
								  const std::shared_ptr<VoxelLocalMap> &voxel_local_map) {
	// std::lock_guard<std::mutex> lock(state_estimator_mutex);
	auto start = std::chrono::steady_clock::now();
	// State tmp_state =  current_state;
	State tmp_state;
	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		tmp_state = current_state;
	}
	std::vector<Eigen::Vector3f> points_world;
	points_world.reserve(points.size());

	for (int iter = 0; iter < max_iteration; iter++) {
		points_world.clear();
		for (auto it : points)
			points_world.push_back(tmp_state.rotation * it + tmp_state.position);
		// auto start_1 = std::chrono::steady_clock::now();
		auto correspondence = voxel_local_map->findCorrespondence(points_world, 1.0);
		// auto middle_1 = std::chrono::steady_clock::now();
		// double elapsed_ms_m1 = std::chrono::duration<double, std::milli>(middle_1 - start_1).count();
		// spdlog::info("[Estimator] getting correspondence took {} ms", elapsed_ms_m1);
		int correspondence_num = static_cast<int>(correspondence.size());
		if (correspondence_num < min_correspondence_threshold) {
			spdlog::info("Insufficient correspondence {}, skipping update.", correspondence_num);
			return tmp_state;
		}
		// auto start_2 = std::chrono::steady_clock::now();
		Eigen::MatrixXf H(correspondence_num, 18);
		Eigen::VectorXf r(correspondence_num);
		Eigen::VectorXf R_inv(correspondence_num);
		Eigen::Matrix<float, 18, 18> G = Eigen::Matrix<float, 18, 18>::Zero();
		std::vector<float> residual_vec(correspondence_num), huber_weights(correspondence_num);
		double variance = 0.0;
		float mean = 0.0;

		for (int i = 0; i < correspondence_num; i++) {
			auto &point = points[correspondence[i].point_idx];
			auto &normal = correspondence[i].normal;
			auto &centroid = correspondence[i].centroid;
			// Computing Jacobian
			Eigen::Matrix3f point_skew = Sophus::SO3f::hat(point);
			H.row(i).setZero();
			H.row(i).block<1, 3>(0, 0) = -normal.transpose() * tmp_state.rotation.matrix() * point_skew;
			H.row(i).block<1, 3>(0, 3) = normal.transpose();
			const float residual = normal.dot(points_world[correspondence[i].point_idx] - centroid);
			residual_vec[i] = residual;
			mean += residual;
			r(i) = residual;
		}
		mean /= correspondence_num;
		for (int i = 0; i < correspondence_num; i++) {
			variance += static_cast<double>((residual_vec[i] - mean) * (residual_vec[i] - mean));
		}
		variance /= correspondence_num;
		float std = std::sqrt(static_cast<float>(variance)) / 3.0f;
		for (int i = 0; i < correspondence_num; i++) {
			residual_vec[i] /= std::max(std, 1e-6f);
			float residual_abs = std::abs(residual_vec[i]);
			if (residual_abs > huber_loss_delta) {
				huber_weights[i] = huber_loss_delta / residual_abs;
			} else {
				huber_weights[i] = 1.0f;
			}
			R_inv(i) = huber_weights[i] / ((lidar_noise_std * lidar_noise_std) + 0.001f);
		}
		// auto middle_2 = std::chrono::steady_clock::now();
		// double elapsed_ms_m2 = std::chrono::duration<double, std::milli>(middle_2 - start_2).count();
		// spdlog::info("[Estimator] before calculating Kalman gain took {} ms", elapsed_ms_m2);
		// auto start_3 = std::chrono::steady_clock::now();
		Eigen::MatrixXf H_9 = H.block(0, 0, correspondence_num, 9);
		Eigen::MatrixXf HT_R_inv(9, correspondence_num);

		for (int i = 0; i < correspondence_num; i++) {
			HT_R_inv.col(i) = H_9.row(i).transpose() * R_inv(i);
		}

		Eigen::Matrix<float, 9, 9> HT_R_inv_H_9 = HT_R_inv * H_9;
		Eigen::Matrix<float, 9, 1> HT_R_inv_r = HT_R_inv * r;
		Eigen::Matrix<float, 18, 18> P_prior = tmp_state.covariance;
		Eigen::Matrix<float, 18, 18> HT_R_inv_H = Eigen::Matrix<float, 18, 18>::Zero();
		HT_R_inv_H.block<9, 9>(0, 0) = HT_R_inv_H_9;
		Eigen::Matrix<float, 18, 18> information = HT_R_inv_H + P_prior.inverse();
		Eigen::Matrix<float, 18, 18> K = information.inverse();
		G.block<18, 9>(0, 0) = K.block<18, 9>(0, 0) * HT_R_inv_H_9;
		// State correction
		Eigen::VectorXf dx = -K.block<18, 9>(0, 0) * HT_R_inv_r;
		// auto middle_3 = std::chrono::steady_clock::now();
		// double elapsed_ms_m3 = std::chrono::duration<double, std::milli>(middle_3 - start_3).count();
		// spdlog::info("[Estimator] after calculating Kalman gain took {} ms", elapsed_ms_m3);
		tmp_state.rotation = tmp_state.rotation * Sophus::SO3f::exp(dx.segment<3>(0));
		tmp_state.position += dx.segment<3>(3);
		tmp_state.velocity += dx.segment<3>(6);
		tmp_state.bias_gyro += dx.segment<3>(9);
		tmp_state.bias_acc += dx.segment<3>(12);
		tmp_state.gravity += dx.segment<3>(15);

		spdlog::info("Number of Correspondence {}, IEKF update iter {}, dx norm: {}", correspondence_num, iter,
					 dx.norm());
		if (dx.norm() < converge_threshold) {
			Eigen::Matrix<float, 18, 18> I18 = Eigen::Matrix<float, 18, 18>::Identity();
			tmp_state.covariance = (I18 - G) * P_prior;
			break;
		}
	}
	// current_state = tmp_state;
	{
		std::lock_guard<std::mutex> lock(state_estimator_mutex);
		current_state = tmp_state;
	}
	auto end = std::chrono::steady_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
	spdlog::info("[Estimator] updating state took {} ms", elapsed_ms);
	return tmp_state;
}

int StateEstimator::getPreintegrationListSize() {
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	return preint_list.size();
}

Sophus::SE3f StateEstimator::getCurrentPose() {
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	return Sophus::SE3f(current_state.rotation, current_state.position);
}

State StateEstimator::getCurrentState() {
	std::lock_guard<std::mutex> lock(state_estimator_mutex);
	return current_state;
}

} // namespace slio