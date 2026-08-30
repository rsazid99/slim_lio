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

VoxelLocalMap::VoxelLocalMap(double _voxel_size, int _max_points_per_voxel, double _min_planarity) {
	voxel_size = _voxel_size;
	max_points_per_voxel = _max_points_per_voxel;
	min_planarity = _min_planarity;
	inv_voxel_size = 1.0 / voxel_size;
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
		v.sum += p.xyz;
		v.count++;
	}

	std::vector<Eigen::Vector3d> downsampled;
	for (auto it : scan_map) {
		double cnt_inv = 1.0 / it.second.count;
		downsampled.push_back(it.second.sum * cnt_inv);
	}

	return downsampled;
}

void VoxelLocalMap::insert(const std::vector<Eigen::Vector3d> &points) {
	for (const auto &p : points) {
		const Eigen::Vector3i c = coordOf(p);
		uint64_t key = encodeKey(c.x(), c.y(), c.z());
		auto &v = voxel_map[key];

		if (v.count >= max_points_per_voxel)
			continue;

		v.count++;
		v.sum += p;
		v.pp_T_sum += p * p.transpose();

		if (v.count < 3)
			continue;

		double inv_n = 1.0 / v.count;
		v.centroid = v.sum * inv_n;
		// Covariance: E[XX^T] - miu*miu^T
		Eigen::Matrix3d cov = v.pp_T_sum * inv_n - (v.centroid * v.centroid.transpose());
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
		if (eig.info() != Eigen::Success) {
			v.valid = false;
			continue;
		}

		Eigen::Vector3d eigenvalues = eig.eigenvalues();
		v.nomal = eig.eigenvectors().col(0).normalized();
		v.planarity = eigenvalues(0) / (eigenvalues(0) + eigenvalues(1) + eigenvalues(2) + 1e-6);
		const double line_ratio = eigenvalues(0) / (eigenvalues(1) + 1e-12);
		v.valid = (v.planarity < min_planarity) && (line_ratio < 0.5); // sets false for line, edge, and blob
	}
}

std::vector<Correspondence> VoxelLocalMap::findCorrespondence(const std::vector<Eigen::Vector3d> &points,
															  double max_distance) {
	std::vector<Correspondence> results;
	results.reserve(points.size());
	if (voxel_map.empty())
		return results;

	constexpr int SEARCH_NEIGHBOUR = 27;

	for (size_t i = 0; i < points.size(); i++) {
		const Eigen::Vector3d &p = points[i];
		const Eigen::Vector3i c = coordOf(p);
		const VoxelData *best = nullptr;
		double best_dist = max_distance;

		for (int n = 0; n < SEARCH_NEIGHBOUR; n++) {
			const uint64_t key = encodeKey(c.x() + NEIGHBOUR_OFFSETS[n][0], c.y() + NEIGHBOUR_OFFSETS[n][1],
										   c.z() + NEIGHBOUR_OFFSETS[n][2]);
			auto it = voxel_map.find(key);
			if (it == voxel_map.end() || !it->second.valid)
				continue;

			const VoxelData &v = it->second;
			const double dist = std::abs(v.nomal.dot(p - v.centroid));
			if (dist < best_dist) {
				best_dist = dist;
				best = &v;
			}
			if (n == 0 && best)
				break;
		}

		if (best) {
			Correspondence corr;
			corr.centroid = best->centroid;
			corr.normal = best->nomal;
			corr.point_idx = static_cast<int>(i);
			results.emplace_back(corr);
		}
	}
	return results;
}

void VoxelLocalMap::evict(const Eigen::Vector3d &pos, double radius) {
	size_t before = voxel_map.size();
	for (auto it = voxel_map.begin(); it != voxel_map.end();) {
		it = ((it->second.sum / it->second.count - pos).norm() > radius) ? voxel_map.erase(it) : std::next(it);
	}
	spdlog::info("[Map] evict: {} -> {} voxels (radius {} m)", before, voxel_map.size(), radius);
}

} // namespace slio