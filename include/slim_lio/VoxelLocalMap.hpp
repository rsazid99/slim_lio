/*
 * @file            include/slim_lio/VoxelMap.hpp
 * @description
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-05-03 14:20:14
 * @lastModified    2026-05-03 14:20:14
 * Copyright ©Sazid Rahman Simanto All rights reserved
 */

#ifndef VOXELLOCALMAP_H
#define VOXELLOCALMAP_H

#include "slim_lio/types.hpp"
#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <nanoflann.hpp>
#include <unordered_map>
namespace slio {

class VoxelLocalMap {
  public:
	VoxelLocalMap(double _voxel_size, int _max_voxel, double _min_planarity);
	~VoxelLocalMap();

	std::vector<Eigen::Vector3d> filterPointCloud(const std::vector<PointCloud> &points);
	void insert(const std::vector<Eigen::Vector3d> &points);
	std::vector<Correspondence> findCorrespondence(const std::vector<Eigen::Vector3d> &points, double max_distance);
	void evict(const Eigen::Vector3d &pos, double radius);

  private:
	static constexpr int OFFSET = 1 << 20;
	static constexpr int NEIGHBOUR_OFFSETS[27][3] = {
		{0, 0, 0},	 {1, 0, 0},	  {-1, 0, 0}, {0, 1, 0},   {0, -1, 0},	{0, 0, 1},	 {0, 0, -1},
		{1, 1, 0},	 {1, -1, 0},  {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1},	{1, 0, -1},	 {-1, 0, 1},
		{-1, 0, -1}, {0, 1, 1},	  {0, 1, -1}, {0, -1, 1},  {0, -1, -1}, {1, 1, 1},	 {1, 1, -1},
		{1, -1, 1},	 {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
	uint64_t encodeKey(int x, int y, int z) const {
		return ((uint64_t)(x + OFFSET) << 42) | ((uint64_t)(y + OFFSET) << 21) | ((uint64_t)(z + OFFSET));
	}
	Eigen::Vector3i coordOf(const Eigen::Vector3d &p) const {
		return Eigen::Vector3i(static_cast<int>(std::floor(p[0] * inv_voxel_size)),
							   static_cast<int>(std::floor(p[1] * inv_voxel_size)),
							   static_cast<int>(std::floor(p[2] * inv_voxel_size)));
	}
	double voxel_size, inv_voxel_size, max_points_per_voxel, min_planarity;
	std::unordered_map<uint64_t, VoxelData> voxel_map;
};

} // namespace slio
#endif