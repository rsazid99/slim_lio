/*
 * @file            src/VoxelMap.cpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-05-03 14:20:25
 * @lastModified    2026-05-03 14:20:25
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#include "slim_lio/VoxelLocalMap.hpp"

namespace slio {

VoxelLocalMap::VoxelLocalMap(double _voxel_size, int _max_voxel, int _min_points, double _min_planarity) {
	voxel_size = _voxel_size;
	max_voxel = _max_voxel;
	min_points = _min_points;
	min_planarity = _min_planarity;
	inv_voxel_size = 1.0 / voxel_size;
	keyframe_empty = true;
	translation_th = 1.0f;
    min_keyframe = 3;
	angle_th = 30.0f * (3.1416f / 180.0f);
	const int buffer_size = 20;
	poses = boost::circular_buffer<Sophus::SE3d>(buffer_size);
	scan_clouds = boost::circular_buffer<std::vector<Eigen::Vector3d>>(buffer_size);
}

VoxelLocalMap::~VoxelLocalMap() {}

std::vector<Eigen::Vector3d> VoxelLocalMap::filterPointCloud(const std::vector<PointCloud> &points) {
	std::unordered_map<uint64_t, VoxelData> scan_map;

	for (auto p : points) {
		int ix = static_cast<int>(std::floor(p.xyz[0] * inv_voxel_size));
		int iy = static_cast<int>(std::floor(p.xyz[1] * inv_voxel_size));
		int iz = static_cast<int>(std::floor(p.xyz[2] * inv_voxel_size));

		uint64_t key = encodeKey(ix, iy, iz);

		auto &v = scan_map[key];
		v.sum += p.xyz.cast<double>();
		v.count++;
	}
	std::vector<Eigen::Vector3d> downsampled;
	for (auto it : scan_map) {
		double cnt_inv = 1.0 / it.second.count;
		downsampled.push_back((it.second.sum * cnt_inv).cast<double>());
	}

	return downsampled;
}

bool VoxelLocalMap::isKeyframe(Sophus::SE3d pose) {
	if (keyframe_empty || scan_clouds.size() < min_keyframe)
		return true;

	Sophus::SE3d dT = last_pose.inverse() * pose;
	spdlog::info("Calculated displacement {}, angle movement {}", dT.translation().norm(), dT.so3().log().norm());
	if (dT.translation().norm() > translation_th || dT.so3().log().norm() > angle_th)
		return true;

	return false;
}

void VoxelLocalMap::addKeyframe(Sophus::SE3d pose, std::vector<Eigen::Vector3d> &points) {
	if (keyframe_empty) {
		keyframe_empty = false;
	}

	poses.push_back(pose);
	last_pose = pose;
	voxel_map.clear();
	local_map.pts.clear();
	local_normals.clear();

	std::vector<Eigen::Vector3d> all_points;
    all_points.reserve(500000);
	scan_clouds.push_back(points);

	for (size_t i = 0; i < scan_clouds.size(); i++) {
		for (auto it : scan_clouds[i])
			all_points.push_back(it);
	}

	for (auto p : all_points) {
		int ix = static_cast<int>(std::floor(p[0] * inv_voxel_size));
		int iy = static_cast<int>(std::floor(p[1] * inv_voxel_size));
		int iz = static_cast<int>(std::floor(p[2] * inv_voxel_size));

		uint64_t key = encodeKey(ix, iy, iz);

		auto &v = voxel_map[key];
		v.sum += p.cast<double>();
		v.pp_T_sum += p.cast<double>() * p.transpose().cast<double>();
		v.count++;
	}
	for (auto &it : voxel_map) {
		double inv_n = 1.0 / it.second.count;
		Eigen::Vector3d centroid = it.second.sum * inv_n;
		// Covariance: E[XX^T] - miu*miu^T
		Eigen::Matrix3d cov = it.second.pp_T_sum * inv_n - (centroid * centroid.transpose());

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
		int min_idx;
		double value = eig.eigenvalues().minCoeff(&min_idx);
		Eigen::Vector3d eigenvalues = eig.eigenvalues().cast<double>();
		it.second.nomal = eig.eigenvectors().col(min_idx).normalized().cast<double>();
		it.second.centroid = (centroid).cast<double>();
		it.second.planarity = value / (eigenvalues(0) + eigenvalues(1) + eigenvalues(2) + 1e-6);
		it.second.valid = (it.second.planarity < min_planarity);
		if (it.second.valid) {
			local_map.pts.push_back(it.second.centroid);
			local_normals.push_back(it.second.nomal);
			// spdlog::info("Point added to keyframe {}, {}, {}", it.second.planarity, min_planarity, it.second.valid);
		}
	}
	kdtree = std::make_unique<KDTree>(3, local_map, nanoflann::KDTreeSingleIndexAdaptorParams(64));
	kdtree->buildIndex();
	// spdlog::info("Point added to keyframe.");
}

std::vector<Correspondence> VoxelLocalMap::findCorrespondence(const std::vector<Eigen::Vector3d> &points,
															  double max_distance) {
	double min_dist_sq = max_distance * max_distance;
	std::vector<Correspondence> results;
	results.reserve(points.size());

	if (local_map.pts.empty())
		return results;

	for (size_t i = 0; i < points.size(); i++) {
		size_t ret_index = 0;
		double dist_sq = std::numeric_limits<double>::max();

		nanoflann::KNNResultSet<double> resultSet(1);
		resultSet.init(&ret_index, &dist_sq);
		const double query_pt[3] = {points[i][0], points[i][1], points[i][2]};

		kdtree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams());

		if (dist_sq < min_dist_sq) {
			Correspondence best;
			best.centroid = local_map.pts[ret_index];
			best.normal = local_normals[ret_index];
			best.point_idx = i;
			results.emplace_back(best);
		}
	}
	return results;
}
std::vector<Eigen::Vector3d> VoxelLocalMap::getLocalMap() { return local_map.pts; }

} // namespace slio