/*
 * @file            src/utils.cpp
 * @description     
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-27 23:38:58
 * @lastModified    2026-11-02 00:43:08
 * Copyright (c) 2026 Sazid Rahman Simanto
 * SPDX-License-Identifier: MIT
*/

#include <algorithm>
#include <slim_lio/utils.hpp>
namespace slio {

void parsePointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg, std::vector<PointCloud> &points) {

	sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
	sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
	sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
	sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");
	sensor_msgs::PointCloud2ConstIterator<double> iter_time(*msg, "timestamp");

	for (; iter_x != iter_x.end();) {
		points.push_back(PointCloud(*iter_x, *iter_y, *iter_z, *iter_intensity, *iter_time * 1e-9));
		++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_time;
	}
}

void ParseLivox(const sensor_msgs::msg::PointCloud2::SharedPtr msg, std::vector<PointCloud> &points) {
	points.reserve(msg->width * msg->height);

	double scan_start_time = rclcpp::Time(msg->header.stamp).seconds();

	sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
	sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
	sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
	sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");
	sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_time(*msg, "offset_time");

	for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_time) {
		double point_time = scan_start_time + static_cast<double>(*iter_time) * 1e-9;

		points.emplace_back(*iter_x, *iter_y, *iter_z, *iter_intensity, point_time);
	}
}

void ParseOuster(const sensor_msgs::msg::PointCloud2::SharedPtr msg, std::vector<PointCloud> &points) {
	points.reserve(msg->width * msg->height);

	double scan_start_time = rclcpp::Time(msg->header.stamp).seconds();

	sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
	sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
	sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
	sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");
	sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_time(*msg, "t"); // ns since scan start

	constexpr double kMinRangeSq = 0.5 * 0.5; // Ouster blind zone / self-returns
	constexpr double kMaxOffset = 0.15;       // t is garbage on no-return points

	for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_time) {
		const float x = *iter_x, y = *iter_y, z = *iter_z;
		if (x == 0.0f && y == 0.0f && z == 0.0f) // no-return points are zero-filled
			continue;
		if (x * x + y * y + z * z < kMinRangeSq)
			continue;
		const double offset = static_cast<double>(*iter_time) * 1e-9;
		if (offset > kMaxOffset)
			continue;

		points.emplace_back(x, y, z, *iter_intensity, scan_start_time + offset);
	}
}

void publishPoseWithPath(nav_msgs::msg::Path &path, const State &state, double timestamp,
						 const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub,
						 const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub) {
	rclcpp::Time ros_time(static_cast<int64_t>(timestamp * 1e9));
	Eigen::Quaterniond q(state.rotation.matrix());
	q.normalize();

	auto pose_msg = geometry_msgs::msg::PoseStamped();
	pose_msg.header.stamp = ros_time;
	pose_msg.header.frame_id = "odom";

	pose_msg.pose.position.x = state.position.x();
	pose_msg.pose.position.y = state.position.y();
	pose_msg.pose.position.z = state.position.z();

	pose_msg.pose.orientation.x = q.x();
	pose_msg.pose.orientation.y = q.y();
	pose_msg.pose.orientation.z = q.z();
	pose_msg.pose.orientation.w = q.w();

	pub->publish(pose_msg);

	path.header.stamp = ros_time;
	path.header.frame_id = "odom";
	path.poses.push_back(pose_msg);

	path_pub->publish(path);
}

void publishOdometry(const State &state, double timestamp, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub,
					 std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broad) {
	rclcpp::Time ros_time(static_cast<int64_t>(timestamp * 1e9));
	Eigen::Quaterniond q(state.rotation.matrix());
	q.normalize();

	auto odom_msg = nav_msgs::msg::Odometry();
	odom_msg.header.stamp = ros_time;
	odom_msg.header.frame_id = "odom";
	odom_msg.child_frame_id = "base_link";

	odom_msg.pose.pose.position.x = state.position.x();
	odom_msg.pose.pose.position.y = state.position.y();
	odom_msg.pose.pose.position.z = state.position.z();

	odom_msg.pose.pose.orientation.x = q.x();
	odom_msg.pose.pose.orientation.y = q.y();
	odom_msg.pose.pose.orientation.z = q.z();
	odom_msg.pose.pose.orientation.w = q.w();

    Eigen::Matrix<double, 6, 6> P;
    P.block<3, 3>(0, 0) = state.covariance.block<3, 3>(3, 3);   // position
    P.block<3, 3>(0, 3) = state.covariance.block<3, 3>(3, 0);   // pos–rot cross terms
    P.block<3, 3>(3, 0) = state.covariance.block<3, 3>(0, 3);
    P.block<3, 3>(3, 3) = state.covariance.block<3, 3>(0, 0);   // rotation
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            odom_msg.pose.covariance[i * 6 + j] = P(i, j);
        }
    }

	odom_msg.twist.twist.linear.x = state.velocity.x();
	odom_msg.twist.twist.linear.y = state.velocity.y();
	odom_msg.twist.twist.linear.z = state.velocity.z();
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            odom_msg.twist.covariance[i * 6 + j] = state.covariance(6 + i, 6 + j);
        }
    }

	static bool firs_frame = true;
	static double last_timestep = 0.0;
	static Eigen::Matrix3d prev_rotation = Eigen::Matrix3d::Identity();

	if (!firs_frame) {
		double dt = timestamp - last_timestep;

		Eigen::Matrix3d dR = prev_rotation.transpose() * state.rotation.matrix();
		Eigen::AngleAxisd angle_axis(dR);
		double angle = angle_axis.angle();
		Eigen::Vector3d axis = angle_axis.axis();
		// omega = Rotation_world_imu * rotation_axis * angular_speed
		Eigen::Vector3d omega_world = (prev_rotation * axis) * (angle / dt);

		odom_msg.twist.twist.angular.x = omega_world.x();
		odom_msg.twist.twist.angular.y = omega_world.y();
		odom_msg.twist.twist.angular.z = omega_world.z();
	} else {
		odom_msg.twist.twist.angular.x = 0;
		odom_msg.twist.twist.angular.y = 0;
		odom_msg.twist.twist.angular.z = 0;
		firs_frame = false;
	}
	prev_rotation = state.rotation.matrix();
	last_timestep = timestamp;

	pub->publish(odom_msg);

	geometry_msgs::msg::TransformStamped transform;
	transform.header.stamp = ros_time;
	transform.header.frame_id = "odom";
	transform.child_frame_id = "base_link";

	transform.transform.translation.x = state.position.x();
	transform.transform.translation.y = state.position.y();
	transform.transform.translation.z = state.position.z();

	transform.transform.rotation.x = q.x();
	transform.transform.rotation.y = q.y();
	transform.transform.rotation.z = q.z();
	transform.transform.rotation.w = q.w();

	tf_broad->sendTransform(transform);
}

void publishCloud(std::vector<Eigen::Vector3d> &cloud, double timestamp,
					 const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub) {
	if (cloud.empty())
		return;
	sensor_msgs::msg::PointCloud2 cloud_msg;
	rclcpp::Time ros_time(static_cast<int64_t>(timestamp * 1e9));
	cloud_msg.header.stamp = ros_time;
	cloud_msg.header.frame_id = "base_link";
	cloud_msg.height = 1;
	cloud_msg.width = cloud.size();
	cloud_msg.is_dense = false;
	cloud_msg.is_bigendian = false;

	sensor_msgs::PointCloud2Modifier mod(cloud_msg);
	mod.setPointCloud2FieldsByString(1, "xyz");
	mod.resize(cloud.size());

	sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
	sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
	sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

	for (const auto &point : cloud) {
		*iter_x = static_cast<float>(point.x());
		*iter_y = static_cast<float>(point.y());
		*iter_z = static_cast<float>(point.z());

		++iter_x;
		++iter_y;
		++iter_z;
	}
	pub->publish(cloud_msg);
}

} // namespace slio