#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <memory>
#include <vector>

class LioNode : public rclcpp::Node
{
public:
    LioNode() : Node("lio_node")
    {   
        // Declare parameters
        this->declare_parameter<std::string>("imu_topic", "livox/imu");
        this->declare_parameter<std::string>("lidar_topic", "livox/lidar");
        this->declare_parameter<std::string>("config_file", "");
        this->declare_parameter<int>("imu_sample_rate", 200);

        // Get parameters
        std::string imu_topic_  = this->get_parameter("imu_topic").as_string();
        std::string lidar_topic_  = this->get_parameter("lidar_topic").as_string();
        std::string config_file_  = this->get_parameter("config_file").as_string();
        int imu_sample_rate_  = this->get_parameter("imu_sample_rate").as_int();

        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
    }

};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LioNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}