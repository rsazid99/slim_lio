/*
 * @file            src/lio_node.cpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-22 19:43:25
 * @lastModified    2026-04-23 02:23:29
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#include <deque>
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
// SLIO Includes
#include "slim_lio/StateEstimator.hpp"
#include "slim_lio/types.hpp"
#include "slim_lio/utils.hpp"
using namespace slio;
class LioNode : public rclcpp::Node {
  public:
    LioNode() : Node("lio_node") {
        // Declare parameters
        this->declare_parameter<std::string>("imu_topic", "/livox/imu");
        this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<int>("init_imu_samples", 200);

        // Get parameters
        std::string imu_topic_ = this->get_parameter("imu_topic").as_string();
        std::string lidar_topic_ = this->get_parameter("lidar_topic").as_string();
        std::string config_file_ = this->get_parameter("config_file").as_string();
        init_imu_samples_ = this->get_parameter("init_imu_samples").as_int();
        gravity_initialized_ = false;
        last_tstamp = 0.0;

        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "IMU samples for gravity initialization %d", init_imu_samples_);

        // Create subscribers
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 1000, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                this->imuCallback(msg);
            });
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, 10, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                this->lidarCallback(msg);
            });
        estimator_ = std::make_shared<StateEstimator>();
    }

  private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        float tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        float imu_dt_ = tstamp - last_tstamp;
        auto imu_data = std::make_shared<IMUData>(
            imu_dt_, tstamp, Eigen::Vector3f(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
            Eigen::Vector3f(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));
        last_tstamp = tstamp;
        // RCLCPP_INFO(this->get_logger(), "gravity Init %d.", gravity_initialized_);
        if (!gravity_initialized_) {
            init_imu_buffer_.push_back(*imu_data);

            if (init_imu_buffer_.size() >= init_imu_samples_) {
                if (estimator_->getGravityInit(init_imu_buffer_)) {
                    RCLCPP_INFO(this->get_logger(), "Successfully initialized gravity.");
                    gravity_initialized_ = true;
                } else {
                    RCLCPP_INFO(this->get_logger(), "Couldn't initialize gravity.");
                }
                init_imu_buffer_.clear();
            }
            return;
        }
        estimator_->propagateIMU(*imu_data);
        if(!propagation_started_) propagation_started_ = true;
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // RCLCPP_INFO(this->get_logger(), "I Found LiDAR Data.");
        if(!gravity_initialized_ || !propagation_started_) return;

        float tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::vector<PointCloud> points;

        parsePointcloud(msg, points);
        estimator_->undistortPointcloud(points);

        // undistortPointcloud(points, intensity, timestamps);
        RCLCPP_INFO(this->get_logger(), " Size of pointcloud %d", points.size());
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

    int init_imu_samples_;
    bool gravity_initialized_;
    bool propagation_started_;
    float last_tstamp;
    std::vector<IMUData> init_imu_buffer_;
    std::shared_ptr<StateEstimator> estimator_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LioNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}