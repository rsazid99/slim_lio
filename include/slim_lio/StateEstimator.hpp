/*
 * @file            include/slim_lio/StateEstimator.hpp
 * @description     
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-25 22:03:43
 * @lastModified    2026-08-30 15:42:43
 * Copyright (c) 2026 Sazid Rahman Simanto
 * SPDX-License-Identifier: MIT
*/

#ifndef STATEESTIMATOR_H
#define STATEESTIMATOR_H

#include "slim_lio/VoxelLocalMap.hpp"
#include "slim_lio/types.hpp"
#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <math.h>
#include <mutex>
#include <sophus/se3.hpp>
#include <vector>
namespace slio {
class StateEstimator {
  public:
	StateEstimator(int _max_iteration, int _min_correspondence_threshold, double _huber_loss_delta,
				   double _converge_threshold, double _lidar_noise_std, double gyro_noise_std,
				   double gyro_bias_noise_std, double acc_noise_std, double acc_bias_noise_std,
				   double gravity_noise_std);
	~StateEstimator();

	bool getGravityInit(const std::vector<IMUData> &imu_buffer);
	State getState();
	void propagateIMU(const IMUData &imu_data);
	void undistortPointcloud(std::vector<PointCloud> &points);
	// IMU-propagated motion from the current state's instant (the deskew reference: last propagated
	// sample <= scan end) to time t, i.e. T_ref^-1 * T_prop(t), integrated exactly as
	// undistortPointcloud does. Right-multiplying the updated pose by it moves the estimate to
	// another instant of the scan (e.g. its middle) consistently with the deskewed cloud.
	Sophus::SE3d getRelativePropagatedMotion(double t);
	State updateState(std::vector<Eigen::Vector3d> &points, const std::shared_ptr<VoxelLocalMap> &voxel_local_map);

  private:
	// Constant-rate integration of propagated sample s to time t (gyro/accel held at s's values).
	static Sophus::SE3d propagatedPose(const StateWithStamp &s, double t);
	State current_state;
	bool g_initialized;
	int max_iteration, min_correspondence_threshold;
	double huber_loss_delta, converge_threshold, lidar_noise_std;
	std::mutex state_estimator_mutex;
	std::deque<StateWithStamp> preint_list;
	Eigen::Matrix<double, 18, 18> process_noise;
};

} // namespace slio
#endif // STATEESTIMATOR_H