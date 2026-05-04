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

#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include "slim_lio/types.hpp"
namespace slio {

class VoxelLocalMap{
public:
    VoxelLocalMap(double _voxel_size, int _max_voxel, int _min_points, double _min_planarity);
    ~VoxelLocalMap();

    std::vector<Eigen::Vector3f> filterPointCloud(const std::vector<Eigen::Vector3f> &points);
    bool isKeyframe(Sophus::SE3f pose);
    void addKeyframe(Sophus::SE3f pose, std::vector<Eigen::Vector3f> points);

private:
    static constexpr int OFFSET = 1 << 20;
    uint64_t encodeKey(int x, int y, int z) const {
        return ((uint64_t)(x + OFFSET) << 42) | ((uint64_t)(y + OFFSET) << 21) | ((uint64_t)(z + OFFSET));
    }
    double voxel_size, inv_voxel_size, min_planarity;
    size_t max_voxel, min_points;
    Sophus::SE3f last_pose;
    float translation_th, angle_th;
    bool keyframe_empty;
    boost::circular_buffer<Sophus::SE3f> poses;
    boost::circular_buffer<std::vector<Eigen::Vector3f>> scan_clouds;
    std::vector<Eigen::Vector3f> downsampled_map;
};

}
#endif