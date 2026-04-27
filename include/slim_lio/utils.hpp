/*
 * @file            include/slim_lio/utils.hpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-23 02:13:40
 * @lastModified    2026-04-23 03:23:16
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#ifndef UTILS_H
#define UTILS_H

#include "slim_lio/types.hpp"
#include <Eigen/Dense>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace slio {

void parsePointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg, std::vector<PointCloud> &points);

// void undistortPointcloud(std::vector<Eigen::Vector3f> &points, std::vector<float> &intensity, std::vector<double> &timestamps);

} // namespace slio
#endif // UTILS_H
