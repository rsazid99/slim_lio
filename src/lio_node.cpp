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
#include "slim_lio/VoxelLocalMap.hpp"
using namespace slio;
class LioNode : public rclcpp::Node {
public:
    LioNode() : Node("lio_node") {
        // Declare parameters
        this->declare_parameter<std::string>("imu_topic", "/livox/imu");
        this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
        this->declare_parameter<int>("init_imu_samples", 200);
        this->declare_parameter<std::vector<double>>("R_il", {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        });
        this->declare_parameter<std::vector<double>>("t_il", {
            0.0, 0.0, 0.0
        });
        this->declare_parameter<int>("voxel_size", 0.20);
        this->declare_parameter<int>("voxel_min_points", 3);
        this->declare_parameter<int>("max_voxel_num", 500000);
        this->declare_parameter<int>("min_planarity", 0.1);
        
        // Get parameters
        std::string imu_topic_ = this->get_parameter("imu_topic").as_string();
        std::string lidar_topic_ = this->get_parameter("lidar_topic").as_string();
        init_imu_samples_ = static_cast<std::size_t>(this->get_parameter("init_imu_samples").as_int());
        auto R_vec = this->get_parameter("R_il").as_double_array();
        Eigen::Matrix3f R_il;
        R_il << static_cast<float>(R_vec[0]), static_cast<float>(R_vec[1]), static_cast<float>(R_vec[2]),
                static_cast<float>(R_vec[3]), static_cast<float>(R_vec[4]), static_cast<float>(R_vec[5]),
                static_cast<float>(R_vec[6]), static_cast<float>(R_vec[7]), static_cast<float>(R_vec[8]);
        auto t_vec = this->get_parameter("t_il").as_double_array();
        Eigen::Vector3f t_il;
        t_il << static_cast<float>(t_vec[0]), static_cast<float>(t_vec[1]), static_cast<float>(t_vec[2]);
        param_voxel_size = this->get_parameter("voxel_size").as_double();
        param_voxel_min_points = this->get_parameter("voxel_min_points").as_int();
        param_max_voxel_num = this->get_parameter("max_voxel_num").as_int();
        param_min_planarity = this->get_parameter("min_planarity").as_double();

        // Initialize local parameters
        gravity_initialized_ = false;
        local_map_initialized_ = false;
        last_tstamp_ = 0.0;
        T_il_ = Sophus::SE3f(R_il, t_il);

        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "IMU samples for gravity initialization %ld", init_imu_samples_);

        // Create subscribers
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 1000, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                this->imuCallback(msg);
            });
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, 10, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                this->lidarCallback(msg);
            });
        // create objects
        estimator_ = std::make_shared<StateEstimator>();
        voxel_local_map_ = std::make_shared<VoxelLocalMap>(param_voxel_size, param_voxel_min_points,
            param_max_voxel_num, param_min_planarity
        );
    }

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        double tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double imu_dt_ = tstamp - last_tstamp_;
        auto imu_data = std::make_shared<IMUData>(
            imu_dt_, tstamp, Eigen::Vector3f(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
            Eigen::Vector3f(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));
        last_tstamp_ = tstamp;
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
        RCLCPP_INFO(this->get_logger(), "Successfully propagated IMU.");
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // RCLCPP_INFO(this->get_logger(), "I Found LiDAR Data.");
        if (!gravity_initialized_ || estimator_->getPreintegrationListSize() < 200)
            return;

        double tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::vector<PointCloud> points;

        parsePointcloud(msg, points);
        RCLCPP_INFO(this->get_logger(), "Successfully parsed LiDAR pointclouds.");
        // Transformed lidar pointcloud from lidar to imu frame
        std::vector<Eigen::Vector3f> points_imu;
        for(auto it: points) {
            points_imu.push_back(T_il_ * it.xyz);
        }
        if(!local_map_initialized_) {
            local_map_initialized_ = true;
            voxel_local_map_->addKeyframe(estimator_->getCurrentPose(), points_imu);
            return;
        }
        estimator_->undistortPointcloud(points);
        if(voxel_local_map_->isKeyframe(estimator_->getCurrentPose())) {
            voxel_local_map_->addKeyframe(estimator_->getCurrentPose(), points_imu);
        }
        // undistortPointcloud(points, intensity, timestamps);
        RCLCPP_INFO(this->get_logger(), " Size of pointcloud %ld", points.size());
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

    std::size_t init_imu_samples_;
    bool gravity_initialized_, local_map_initialized_;
    double last_tstamp_;
    double param_voxel_size, param_min_planarity;
    int param_voxel_min_points, param_max_voxel_num;
    std::vector<IMUData> init_imu_buffer_;
    std::shared_ptr<StateEstimator> estimator_;
    std::shared_ptr<VoxelLocalMap> voxel_local_map_;
    Sophus::SE3f T_il_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LioNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}