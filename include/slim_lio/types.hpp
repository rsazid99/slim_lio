/*
 * @file            include/slim_lio/types.hpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-26 02:08:53
 * @lastModified    2026-04-26 02:10:01
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <math.h>
#include <nanoflann.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace slio {

struct IMUData {
	double dt, timestamp;
	Eigen::Vector3d gyro, acc;

	IMUData() : dt(0), timestamp(0), gyro(Eigen::Vector3d::Zero()), acc(Eigen::Vector3d::Zero()) {}

	IMUData(double _dt, double _timestamp, const Eigen::Vector3d _gyro, const Eigen::Vector3d _acc) {
		dt = _dt;
		timestamp = _timestamp;
		gyro = _gyro;
		acc = _acc;
	}
};

struct PointCloud {
	double intensity;
	double timestamp;
	Eigen::Vector3d xyz;

	PointCloud(double _x, double _y, double _z, double _intensity, double _timestamp) {
		xyz = Eigen::Vector3d(_x, _y, _z);
		intensity = _intensity, timestamp = _timestamp;
	}
};

struct State {
	Eigen::Vector3d position, velocity, bias_gyro, bias_acc, gravity;
	Sophus::SO3d rotation;

	Eigen::Matrix<double, 18, 18> covariance;

	State() {
		rotation = Sophus::SO3d(Eigen::Matrix3d::Identity());
		position = velocity = bias_gyro = bias_acc = Eigen::Vector3d::Zero();
		gravity = Eigen::Vector3d(0, 0, -9.81);
		covariance = Eigen::Matrix<double, 18, 18>::Identity() * 0.001;
	}
};

struct StateWithStamp {
	Eigen::Vector3d position, velocity, gyro, accel, gravity;
	Sophus::SO3d rotation;
	Sophus::SE3d pred_pose;
	double timestamp;

	StateWithStamp(double _tstamp, Eigen::Vector3d _gyro, Eigen::Vector3d _accel, Sophus::SO3d _rot,
				   Eigen::Vector3d _pos, Eigen::Vector3d _vel, Eigen::Vector3d _gravity, Sophus::SE3d _pred_pose) {
		timestamp = _tstamp;
		gyro = _gyro;
		accel = _accel;
		rotation = _rot;
		position = _pos;
		velocity = _vel;
		gravity = _gravity;
		pred_pose = _pred_pose;
	}
};

struct VoxelData {
	Eigen::Vector3d sum = Eigen::Vector3d::Zero();
	Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
	Eigen::Matrix3d pp_T_sum = Eigen::Matrix3d::Zero();
	int count = 0;
	bool valid = false;
	Eigen::Vector3d nomal = Eigen::Vector3d::Zero();
	double planarity = 0.0;
};
struct KDPointCloud {
	std::vector<Eigen::Vector3d> pts;

	KDPointCloud() = default;

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }

	template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KDPointCloud>, KDPointCloud, 3>;

struct Correspondence {
	Eigen::Vector3d centroid, normal;
	int point_idx;
};

enum SensorType {
    IMU,
    LIDAR
};
struct SensorData {
    SensorType sensor;
    double timestamp;
    IMUData imudata;
    std::vector<PointCloud> lidardata;
};

struct SensorDataCompare {
	bool operator()(const SensorData &a, const SensorData &b) const {
		if (a.timestamp != b.timestamp)
			return a.timestamp > b.timestamp;      // earliest timestamp first
		return a.sensor == LIDAR && b.sensor == IMU; // tie-break: IMU before lidar
	}
};

} // namespace slio

#endif // TYPES_H