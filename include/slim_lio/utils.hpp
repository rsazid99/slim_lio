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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sophus/se3.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace slio {

void parsePointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg, std::vector<PointCloud> &points);

void publishPose(const State& state, double timestamp, const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub);

void publishOdometry(const State& state, double timestamp, const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub, 
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broad);

void publishLocalMap(std::vector<Eigen::Vector3f>& lmap, double timestamp, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub);

} // namespace slio
#endif // UTILS_H
