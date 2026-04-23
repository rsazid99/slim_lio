/*
 * @file            src/lio_node.cpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-22 19:43:25
 * @lastModified    2026-04-23 02:23:29
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <vector>

class LioNode : public rclcpp::Node {
  public:
    LioNode() : Node("lio_node") {
        // Declare parameters
        this->declare_parameter<std::string>("imu_topic", "/livox/imu");
        this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<int>("imu_sample_rate", 200);

        // Get parameters
        std::string imu_topic_ = this->get_parameter("imu_topic").as_string();
        std::string lidar_topic_ = this->get_parameter("lidar_topic").as_string();
        std::string config_file_ = this->get_parameter("config_file").as_string();
        imu_sample_rate_ = this->get_parameter("imu_sample_rate").as_int();

        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "IMU sample rate: %d Hz", imu_sample_rate_);

        // Create subscribers
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 1000, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                this->imuCallback(msg);
            });
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, 10, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                this->lidarCallback(msg);
            });
    }

  private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "I Found IMU Data.");
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "I Found LiDAR Data.");
    }

    // Subcriber
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // Publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_scan_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_cloud_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;

    // TF broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    int imu_sample_rate_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LioNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}