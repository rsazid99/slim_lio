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

#include "slim_lio/types.hpp"
#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <algorithm>
#include <deque>
#include <iostream>
#include <math.h>
#include <mutex>
#include <sophus/se3.hpp>
#include <vector>
#include "slim_lio/VoxelLocalMap.hpp"
namespace slio {
class StateEstimator {
  public:
    StateEstimator();
    ~StateEstimator();

    bool getGravityInit(const std::vector<IMUData>& imu_buffer);
    void propagateIMU(const IMUData& imu_data);
    void undistortPointcloud(std::vector<PointCloud>& points);
    Eigen::Matrix<float, 1, 18> computeJacobian(const Eigen::Vector3f& point, const Eigen::Vector3f normal);
    void updateState(std::vector<Eigen::Vector3f>& points, const std::shared_ptr<VoxelLocalMap>& voxel_local_map);
    int getPreintegrationListSize();
    Sophus::SE3f getCurrentPose();

  private:
    State current_state;
    bool g_initialized;
    int max_iteration, min_correspondence_threshold;
    float converge_threshold;
    std::mutex state_estimator_mutex;
    std::deque<StateWithStamp> preint_list;
};

} // namespace slio
#endif // STATEESTIMATOR_H