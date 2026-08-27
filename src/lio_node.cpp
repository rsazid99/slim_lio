/*
 * @file            src/lio_node.cpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-22 19:43:25
 * @lastModified    2026-04-23 02:23:29
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#include <condition_variable>
#include <deque>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <vector>
// SLIO Includes
#include "slim_lio/StateEstimator.hpp"
#include "slim_lio/VoxelLocalMap.hpp"
#include "slim_lio/types.hpp"
#include "slim_lio/utils.hpp"
using namespace slio;
class LioNode : public rclcpp::Node {
  public:
	LioNode() : Node("lio_node") {
		// Declare parameters
		this->declare_parameter<std::string>("imu_topic", "/livox/imu");
		this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
		this->declare_parameter<int>("init_imu_samples", 200);
		this->declare_parameter<std::vector<double>>("R_il", {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
		this->declare_parameter<std::vector<double>>("t_il", {0.0, 0.0, 0.0});
		this->declare_parameter<double>("acc_scale", 9.8);
		this->declare_parameter<double>("voxel_size", 0.1);
		this->declare_parameter<int>("max_voxel_num", 500000);
		this->declare_parameter<double>("min_planarity", 0.2);
		this->declare_parameter<double>("gyro_noise_std", 0.001);
		this->declare_parameter<double>("acc_noise_std", 0.001);
		this->declare_parameter<double>("gyro_bias_noise_std", 0.001);
		this->declare_parameter<double>("acc_bias_noise_std", 0.001);
		this->declare_parameter<double>("gravity_noise_std", 0.001);

		// Get parameters
		std::string imu_topic_ = this->get_parameter("imu_topic").as_string();
		std::string lidar_topic_ = this->get_parameter("lidar_topic").as_string();
		init_imu_samples_ = static_cast<std::size_t>(this->get_parameter("init_imu_samples").as_int());
		auto R_vec = this->get_parameter("R_il").as_double_array();
		Eigen::Matrix3d R_il;
		R_il << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5], R_vec[6], R_vec[7], R_vec[8];
		auto t_vec = this->get_parameter("t_il").as_double_array();
		Eigen::Vector3d t_il;
		t_il << t_vec[0], t_vec[1], t_vec[2];
		acc_scale_ = this->get_parameter("acc_scale").as_double();
		param_voxel_size = this->get_parameter("voxel_size").as_double();
		param_max_voxel_num = this->get_parameter("max_voxel_num").as_int();
		param_min_planarity = this->get_parameter("min_planarity").as_double();
		param_gyro_noise_std = this->get_parameter("gyro_noise_std").as_double();
		param_gyro_bias_noise_std = this->get_parameter("gyro_bias_noise_std").as_double();
		param_acc_noise_std = this->get_parameter("acc_noise_std").as_double();
		param_acc_bias_noise_std = this->get_parameter("acc_bias_noise_std").as_double();
		param_gravity_noise_std = this->get_parameter("gravity_noise_std").as_double();

		// Initialize local parameters
		gravity_initialized_ = false;
		local_map_initialized_ = false;
		running_ = true;
		last_tstamp_ = 0.0;
		T_il_ = Sophus::SE3d(R_il, t_il);

		RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic_.c_str());
		RCLCPP_INFO(this->get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
		RCLCPP_INFO(this->get_logger(), "IMU samples for gravity initialization %ld", init_imu_samples_);

		// Create subscribers
		imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
			imu_topic_, 1000, [this](const sensor_msgs::msg::Imu::SharedPtr msg) { this->imuCallback(msg); });
		lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
			lidar_topic_, 20, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { this->lidarCallback(msg); });
		// Create publisher
		odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slim_lio/Odometry", 15);
		pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slim_lio/pose", 100);
		aligned_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slim_lio/aligned_scan", 15);
		map_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slim_lio/map", 15);
		tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
		trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>("/slim_lio/trajectory", 15);

		// create objects
		estimator_ =
			std::make_shared<StateEstimator>(param_gyro_noise_std, param_gyro_bias_noise_std, param_acc_noise_std,
											 param_acc_bias_noise_std, param_gravity_noise_std);
		voxel_local_map_ = std::make_shared<VoxelLocalMap>(param_voxel_size, param_max_voxel_num, param_min_planarity);
		data_processing_thread_ = std::thread(&LioNode::processDataLoop, this);
	}
	~LioNode() {
		running_ = false;
		queue_cv_.notify_all();
		if (data_processing_thread_.joinable()) {
			data_processing_thread_.join();
			RCLCPP_INFO(this->get_logger(), "Data processing thread stopped");
		}
	}

  private:
	void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
		double tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
		double imu_dt_ = tstamp - last_tstamp_;
		auto imu_data = std::make_shared<IMUData>(
			imu_dt_, tstamp, Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
			Eigen::Vector3d(msg->linear_acceleration.x * acc_scale_, msg->linear_acceleration.y * acc_scale_,
							msg->linear_acceleration.z * acc_scale_));
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
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			sensor_data_queue_.push({IMU, tstamp, *imu_data, {}});
			latest_imu_tstamp_ = tstamp;
		}
		queue_cv_.notify_one();
		// RCLCPP_INFO(this->get_logger(), "Successfully propagated IMU.");
	}

	void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
		if (!gravity_initialized_)
			return;
		double tstamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
		std::vector<PointCloud> points;
		ParseLivox(msg, points);
		if (points.empty())
			return;
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			sensor_data_queue_.push({LIDAR, tstamp, IMUData(), std::move(points)});
		}
		queue_cv_.notify_one();
	}

	void processDataLoop() {
		while (running_) {
			SensorData d;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				queue_cv_.wait(lock, [this] {
					if (!running_)
						return true;
					if (sensor_data_queue_.empty())
						return false;
					const SensorData &top = sensor_data_queue_.top();
					// Hold a scan back until IMU coverage reaches its last point,
					// so propagation always precedes the scan's update.
					if (top.sensor == LIDAR)
						return latest_imu_tstamp_ >= top.lidardata.back().timestamp;
					return true;
				});
				if (!running_)
					break;

				d = std::move(const_cast<SensorData &>(sensor_data_queue_.top()));
				sensor_data_queue_.pop();
			}

			if (d.sensor == IMU) {
				estimator_->propagateIMU(d.imudata);
			} else if (d.sensor == LIDAR) {
				auto start = std::chrono::steady_clock::now();
				for (auto &it : d.lidardata) {
					it.xyz = T_il_ * it.xyz;
				}
				if (!local_map_initialized_) {
					local_map_initialized_ = true;
					std::vector<Eigen::Vector3d> init_cloud(d.lidardata.size());
					for (size_t i = 0; i < d.lidardata.size(); i++) {
						init_cloud[i] = d.lidardata[i].xyz;
					}
					voxel_local_map_->addKeyframe(estimator_->getCurrentPose(), init_cloud);
					continue;
				}
				estimator_->undistortPointcloud(d.lidardata);
				std::vector<Eigen::Vector3d> downsampled_points = voxel_local_map_->filterPointCloud(d.lidardata);
				State state = estimator_->updateState(downsampled_points, voxel_local_map_);
				// Publish pose and odometry
				std::vector<Eigen::Vector3d> aligned_cloud(d.lidardata.size());
				for (size_t i = 0; i < d.lidardata.size(); i++) {
					aligned_cloud[i] = state.rotation * d.lidardata[i].xyz + state.position;
				}
				publishPoseWithPath(trajectory_msg_, state, d.timestamp, pose_pub_, trajectory_pub_);
				publishOdometry(state, d.timestamp, odom_pub_, tf_broadcaster_);

				if (voxel_local_map_->isKeyframe(Sophus::SE3d(state.rotation, state.position))) {
					voxel_local_map_->addKeyframe(Sophus::SE3d(state.rotation, state.position), aligned_cloud);
					RCLCPP_INFO(this->get_logger(), "Keyframe added.");
				}
                //std::vector<Eigen::Vector3d> local_map_ = voxel_local_map_->getLocalMap();
			    //publishLocalMap(local_map_, d.timestamp, map_cloud_pub_);
                publishCloud(aligned_cloud, d.timestamp, aligned_cloud_pub_);
				auto end = std::chrono::steady_clock::now();
				double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
				spdlog::info("[LidarCallback] time took {} ms", elapsed_ms);
			}
		}
	}

	// Subcriber
	rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
	rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

	// Publisher
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_cloud_pub_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_cloud_pub_;
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
	nav_msgs::msg::Path trajectory_msg_;

	// TF broadcaster
	std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

	std::size_t init_imu_samples_;
	bool gravity_initialized_, local_map_initialized_;
	double last_tstamp_;
	double acc_scale_, param_voxel_size, param_min_planarity;
	int param_voxel_min_points, param_max_voxel_num;
	double param_gyro_noise_std, param_gyro_bias_noise_std, param_acc_noise_std, param_acc_bias_noise_std,
		param_gravity_noise_std;
	std::vector<IMUData> init_imu_buffer_;
	std::shared_ptr<StateEstimator> estimator_;
	std::shared_ptr<VoxelLocalMap> voxel_local_map_;
	Sophus::SE3d T_il_;
	std::thread data_processing_thread_;
	std::mutex queue_mutex;
	std::condition_variable queue_cv_;
	double latest_imu_tstamp_ = 0.0;
	std::atomic<bool> running_;
	std::priority_queue<SensorData, std::vector<SensorData>, SensorDataCompare> sensor_data_queue_;
};

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<LioNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();

	return 0;
}