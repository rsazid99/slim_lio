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
	bool isKeyframe(Sophus::SE3d pose);
	void addKeyframe(Sophus::SE3d pose, std::vector<Eigen::Vector3d> &points);
	std::vector<Correspondence> findCorrespondence(const std::vector<Eigen::Vector3d> &points, double max_distance);
	std::vector<Eigen::Vector3d> getLocalMap();

  private:
	static constexpr int OFFSET = 1 << 20;
	uint64_t encodeKey(int x, int y, int z) const {
		return ((uint64_t)(x + OFFSET) << 42) | ((uint64_t)(y + OFFSET) << 21) | ((uint64_t)(z + OFFSET));
	}
	double voxel_size, inv_voxel_size, min_planarity;
	size_t max_voxel, min_keyframe;
	Sophus::SE3d last_pose;
	double translation_th, angle_th;
	bool keyframe_empty;
	boost::circular_buffer<Sophus::SE3d> poses;
	boost::circular_buffer<std::vector<Eigen::Vector3d>> scan_clouds;
	std::unordered_map<uint64_t, VoxelData> voxel_map;
	std::unique_ptr<KDTree> kdtree;
	KDPointCloud local_map;
	std::vector<Eigen::Vector3d> local_normals;
};

} // namespace slio
#endif