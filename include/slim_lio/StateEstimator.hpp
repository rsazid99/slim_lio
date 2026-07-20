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
	StateEstimator(float gyro_noise_std, float gyro_bias_noise_std, float acc_noise_std, float acc_bias_noise_std,
				   float gravity_noise_std);
	~StateEstimator();

	bool getGravityInit(const std::vector<IMUData> &imu_buffer);
	void propagateIMU(const IMUData &imu_data);
	void undistortPointcloud(std::vector<PointCloud> &points);
	State updateState(std::vector<Eigen::Vector3f> &points, const std::shared_ptr<VoxelLocalMap> &voxel_local_map);
	int getPreintegrationListSize();
	Sophus::SE3f getCurrentPose();
	State getCurrentState();

  private:
	State current_state, imu_state;
	bool g_initialized;
	int max_iteration, min_correspondence_threshold;
	float huber_loss_delta, converge_threshold, lidar_noise_std;
	std::mutex state_estimator_mutex;
	std::deque<StateWithStamp> preint_list;
	Eigen::Matrix<float, 18, 18> process_noise;
};

} // namespace slio
#endif // STATEESTIMATOR_H