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
	Eigen::Vector3f gyro, acc;

	IMUData(double _dt, double _timestamp, const Eigen::Vector3f _gyro, const Eigen::Vector3f _acc) {
		dt = _dt;
		timestamp = _timestamp;
		gyro = _gyro;
		acc = _acc;
	}
};

struct PointCloud {
	float intensity;
	double timestamp;
	Eigen::Vector3f xyz;

	PointCloud(float _x, float _y, float _z, float _intensity, double _timestamp) {
		xyz = Eigen::Vector3f(_x, _y, _z);
		intensity = _intensity, timestamp = _timestamp;
	}
};

struct State {
	Eigen::Vector3f position, velocity, bias_gyro, bias_acc, gravity;
	Sophus::SO3f rotation;

	Eigen::Matrix<float, 18, 18> covariance;

	State() {
		rotation = Sophus::SO3f(Eigen::Matrix3f::Identity());
		position = velocity = bias_gyro = bias_acc = Eigen::Vector3f::Zero();
		gravity = Eigen::Vector3f(0, 0, -9.81);
		covariance = Eigen::Matrix<float, 18, 18>::Identity() * 0.001;
	}
};

struct StateWithStamp {
	Eigen::Vector3f position, velocity, gyro, accel, gravity;
	Sophus::SO3f rotation;
	Sophus::SE3f pred_pose;
	double timestamp;

	StateWithStamp(double _tstamp, Eigen::Vector3f _gyro, Eigen::Vector3f _accel, Sophus::SO3f _rot,
				   Eigen::Vector3f _pos, Eigen::Vector3f _vel, Eigen::Vector3f _gravity, Sophus::SE3f _pred_pose) {
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
	Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
	Eigen::Matrix3d pp_T_sum = Eigen::Matrix3d::Zero();
	int count = 0;
	bool valid = false;
	Eigen::Vector3f nomal = Eigen::Vector3f::Zero();
	float planarity = 0.0;
};
struct KDPointCloud {
	std::vector<Eigen::Vector3f> pts;

	KDPointCloud() = default;

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline float kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }

	template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KDPointCloud>, KDPointCloud, 3>;

struct Correspondence {
	Eigen::Vector3f centroid, normal;
	int point_idx;
};

} // namespace slio

#endif // TYPES_H