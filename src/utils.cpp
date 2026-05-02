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

}