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
		acc_scale_ = this->declare_parameter<double>("acc_scale", 9.8);
		// Voxel Local Map
		const auto param_voxel_size = this->declare_parameter<double>("voxel_size", 0.5);
		const auto param_max_points_per_voxel = this->declare_parameter<int>("max_points_per_voxel", 30);
		const auto param_min_planarity = this->declare_parameter<double>("min_planarity", 0.1);
		voxel_local_map_ =
			std::make_shared<VoxelLocalMap>(param_voxel_size, param_max_points_per_voxel, param_min_planarity);
		// State Estimator
		const auto param_max_iteration = this->declare_parameter<int>("max_iteration", 10);
		const auto param_min_correspondence_threshold =
			this->declare_parameter<int>("min_correspondence_threshold", 20);
		const auto param_huber_loss_delta = this->declare_parameter<double>("huber_loss_delta", 1.345);
		const auto param_converge_threshold = this->declare_parameter<double>("converge_threshold", 0.01);
		const auto param_lidar_noise_std = this->declare_parameter<double>("lidar_noise_std", 0.05);
		const auto param_gyro_noise_std = this->declare_parameter<double>("gyro_noise_std", 0.001);
		const auto param_acc_noise_std = this->declare_parameter<double>("acc_noise_std", 0.001);
		const auto param_gyro_bias_noise_std = this->declare_parameter<double>("gyro_bias_noise_std", 0.001);
		const auto param_acc_bias_noise_std = this->declare_parameter<double>("acc_bias_noise_std", 0.001);
		const auto param_gravity_noise_std = this->declare_parameter<double>("gravity_noise_std", 0.001);
		estimator_ = std::make_shared<StateEstimator>(
			param_max_iteration, param_min_correspondence_threshold, param_huber_loss_delta, param_converge_threshold,
			param_lidar_noise_std, param_gyro_noise_std, param_gyro_bias_noise_std, param_acc_noise_std,
			param_acc_bias_noise_std, param_gravity_noise_std);

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

		// Initialize local parameters
		gravity_initialized_ = false;
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

		// Processing thread
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
	}

	void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
		if (!gravity_initialized_)
			return;
		std::vector<PointCloud> points;
		ParseLivox(msg, points);
		if (points.empty())
			return;
		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			sensor_data_queue_.push({LIDAR, points.back().timestamp, IMUData(), std::move(points)});
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
				estimator_->undistortPointcloud(d.lidardata);
				std::vector<Eigen::Vector3d> downsampled_points = voxel_local_map_->filterPointCloud(d.lidardata);
				State state = estimator_->updateState(downsampled_points, voxel_local_map_);
				
				std::vector<Eigen::Vector3d> aligned_cloud(d.lidardata.size());
				for (size_t i = 0; i < d.lidardata.size(); i++) {
					aligned_cloud[i] = state.rotation * d.lidardata[i].xyz + state.position;
				}

                // Publish pose and odometry
				publishPoseWithPath(trajectory_msg_, state, d.timestamp, pose_pub_, trajectory_pub_);
				publishOdometry(state, d.timestamp, odom_pub_, tf_broadcaster_);
				voxel_local_map_->insert(downsampled_points);

				if (!evict_initialized_) {
					last_evict_pos_ = state.position;
					evict_initialized_ = true;
				} else if ((state.position - last_evict_pos_).norm() > evict_every_m_) {
					voxel_local_map_->evict(state.position, evict_radius_m_);
					last_evict_pos_ = state.position;
				}

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
	bool gravity_initialized_;
	double last_tstamp_;
	double acc_scale_;
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
	Eigen::Vector3d last_evict_pos_ = Eigen::Vector3d::Zero();
	bool evict_initialized_ = false;
	double evict_every_m_ = 10.0;	// distance between evictions
	double evict_radius_m_ = 150.0; // keep voxels within this radius of the pose
};

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<LioNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();

	return 0;
}