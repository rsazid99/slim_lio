/*
 * @file            src/slam_node.cpp
 * @description     
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-08-30 22:40:55
 * @lastModified    2026-11-01 19:24:49
 * Copyright (c) 2026 Sazid Rahman Simanto
 * SPDX-License-Identifier: MIT
*/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <iomanip>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <unordered_map>
#include <vector>
// GTSAM Includes
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
// SLIO Includes
#include "slim_lio/types.hpp"
using namespace slio;

using std::placeholders::_1;
using std::placeholders::_2;

namespace {

Sophus::SE3d toSE3(const geometry_msgs::msg::PoseWithCovariance &pose) {
	const auto &p = pose.pose.position;
	const auto &q = pose.pose.orientation;
	return Sophus::SE3d(Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized(), Eigen::Vector3d(p.x, p.y, p.z));
}

gtsam::Pose3 toGtsam(const Sophus::SE3d &T) {
	return gtsam::Pose3(gtsam::Rot3(T.unit_quaternion()), gtsam::Point3(T.translation()));
}

Sophus::SE3d fromGtsam(const gtsam::Pose3 &P) {
	return Sophus::SE3d(Eigen::Quaterniond(P.rotation().matrix()).normalized(), P.translation());
}

std::vector<Eigen::Vector3d> fromPointCloud2(const sensor_msgs::msg::PointCloud2 &msg) {
	std::vector<Eigen::Vector3d> points;
	points.reserve(static_cast<size_t>(msg.width) * msg.height);
	sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z");
	for (; ix != ix.end(); ++ix, ++iy, ++iz)
		points.emplace_back(*ix, *iy, *iz);
	return points;
}

std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d> &points, double leaf) {
	struct Acc {
		Eigen::Vector3d sum = Eigen::Vector3d::Zero();
		int count = 0;
	};
	constexpr int OFFSET = 1 << 20;
	const double inv = 1.0 / leaf;
	std::unordered_map<uint64_t, Acc> grid;
	for (const auto &p : points) {
		uint64_t kx = static_cast<uint64_t>(static_cast<int>(std::floor(p.x() * inv)) + OFFSET);
		uint64_t ky = static_cast<uint64_t>(static_cast<int>(std::floor(p.y() * inv)) + OFFSET);
		uint64_t kz = static_cast<uint64_t>(static_cast<int>(std::floor(p.z() * inv)) + OFFSET);
		auto &v = grid[(kx << 42) | (ky << 21) | kz];
		v.sum += p;
		v.count++;
	}
	std::vector<Eigen::Vector3d> out;
	out.reserve(grid.size());
	for (const auto &it : grid)
		out.push_back(it.second.sum / it.second.count);
	return out;
}

bool writePly(const std::string &path, const std::vector<Eigen::Vector3d> &points) {
	std::ofstream f(path, std::ios::binary);
	if (!f)
		return false;
	f << "ply\nformat binary_little_endian 1.0\nelement vertex " << points.size()
	  << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
	for (const auto &p : points) {
		float xyz[3] = {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
		f.write(reinterpret_cast<const char *>(xyz), sizeof(xyz));
	}
	return f.good();
}

bool writeTum(const std::string &path, const std::vector<Keyframe> &keyframes) {
	std::ofstream f(path);
	if (!f)
		return false;
	f << std::fixed << std::setprecision(9);
	for (const auto &k : keyframes) {
		const Eigen::Vector3d &t = k.T_opt.translation();
		Eigen::Quaterniond q = k.T_opt.unit_quaternion();
		f << k.stamp << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " " << q.z()
		  << " " << q.w() << "\n";
	}
	return f.good();
}

// Refine the world pose T of the body-frame cloud src against the world-frame target
// with point-to-point Gauss-Newton ICP. Accepts only tight, high-overlap alignments.
bool alignICP(const std::vector<Eigen::Vector3d> &src, const KDCloud &target, const KDTree &tree, Sophus::SE3d &T,
			  double &rmse, double &inlier_ratio, size_t max_pts = 1500) {
	const size_t stride = std::max<size_t>(1, src.size() / max_pts);
	double max_corr = 3.0; // generous at first (odometry drift), shrinks as the fit tightens
	const double huber = 0.5;
	for (int iter = 0; iter < 30; ++iter) {
		Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
		Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
		int n = 0;
		for (size_t i = 0; i < src.size(); i += stride) {
			Eigen::Vector3d pw = T * src[i];
			size_t idx;
			double sq_dist;
			if (tree.knnSearch(pw.data(), 1, &idx, &sq_dist) == 0 || sq_dist > max_corr * max_corr)
				continue;
			Eigen::Vector3d r = pw - target.pts[idx];
			double d = std::sqrt(sq_dist);
			double w = d < huber ? 1.0 : huber / d;
			Eigen::Matrix<double, 3, 6> J;
			J.leftCols<3>() = Eigen::Matrix3d::Identity();
			J.rightCols<3>() = -Sophus::SO3d::hat(pw);
			H += w * J.transpose() * J;
			b += w * J.transpose() * r;
			++n;
		}
		if (n < 50)
			return false;
		Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(b);
		T = Sophus::SE3d::exp(dx) * T;
		max_corr = std::max(1.0, max_corr * 0.9);
		if (dx.norm() < 1e-4)
			break;
	}

	const double gate = 0.5;
	int total = 0, inliers = 0;
	double sq_sum = 0.0;
	for (size_t i = 0; i < src.size(); i += stride) {
		Eigen::Vector3d pw = T * src[i];
		size_t idx;
		double sq_dist;
		if (tree.knnSearch(pw.data(), 1, &idx, &sq_dist) == 0)
			continue;
		++total;
		if (sq_dist < gate * gate) {
			++inliers;
			sq_sum += sq_dist;
		}
	}
	if (total == 0 || inliers < 50)
		return false;
	rmse = std::sqrt(sq_sum / inliers);
	inlier_ratio = static_cast<double>(inliers) / total;
	return inlier_ratio > 0.6 && rmse < 0.3;
}

gtsam::ISAM2Params makeIsamParams() {
	gtsam::ISAM2Params params;
	params.relinearizeThreshold = 0.1;
	params.relinearizeSkip = 1;
	return params;
}

} // namespace

class SlamNode : public rclcpp::Node {
  public:
	SlamNode() : Node("slam_node") {
		// Create subscribers
		pose_sub_.subscribe(this, "/slim_lio/Odometry", rclcpp::QoS(20).get_rmw_qos_profile());
		scan_sub_.subscribe(this, "/slim_lio/downsampled_scan", rclcpp::QoS(20).get_rmw_qos_profile());
		sync_ = std::make_shared<Sync>(SyncPolicy(10), pose_sub_, scan_sub_);
		sync_->registerCallback(std::bind(&SlamNode::poseWithScanCallback, this, _1, _2));

		path_pub_ = create_publisher<nav_msgs::msg::Path>("/slim_lio/optimized_path", 10);
		tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

		save_srv_ = create_service<std_srvs::srv::Trigger>(
			"~/save_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
								 std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
				waitForQueueDrain(); // queued keyframes are not in keyframes_ yet
				std::vector<Eigen::Vector3d> map;
				bool tum_ok;
				{
					std::lock_guard<std::mutex> lk(kf_mutex_);
					for (const auto &k : keyframes_)
						for (const auto &p : k.cloud_body)
							map.push_back(k.T_opt * p);
					tum_ok = writeTum("slim_lio_poses.tum", keyframes_);
				}
				map = voxelDownsample(map, 0.2);
				bool ply_ok = writePly("slim_lio_map.ply", map);
				res->success = ply_ok && tum_ok;
				res->message = res->success ? "saved slim_lio_map.ply (" + std::to_string(map.size()) +
												  " points) and slim_lio_poses.tum"
											: "failed to write map/poses";
			});

		loop_thread_ = std::thread(&SlamNode::loopClosureLoop, this);
	}
	~SlamNode() {
		running_ = false;
		queue_cv_.notify_all();
		if (loop_thread_.joinable())
			loop_thread_.join();
	}

  private:
	void poseWithScanCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &odom,
							  const sensor_msgs::msg::PointCloud2::ConstSharedPtr &scan) {
		// Re-broadcast the latest map->odom correction at odometry rate and stamp, so TF
		// consumers (RViz, navigation) can transform every scan; the correction itself
		// only changes at keyframe rate and is refreshed by the optimizer thread.
		publishMapToOdom(rclcpp::Time(odom->header.stamp).seconds());
		Sophus::SE3d T = toSE3(odom->pose);
		if (has_kf_) {
			Sophus::SE3d dT = last_kf_pose_.inverse() * T;
			if (dT.translation().norm() < 1.5 && dT.so3().log().norm() < 0.26)
				return; // 1.5 m / 15°
		}
		last_kf_pose_ = T;
		has_kf_ = true;

		Keyframe kf;
		kf.T_odom = kf.T_opt = T;
		kf.stamp = rclcpp::Time(odom->header.stamp).seconds();
		kf.cloud_body = fromPointCloud2(*scan); // scan arrives in body frame (base_link)

		{
			std::lock_guard<std::mutex> lk(queue_mutex_);
			kf.id = next_id_++;
			kf_queue_.push_back(std::move(kf));
		}
		queue_cv_.notify_one();
	}

	void loopClosureLoop() {
		while (running_) {
			Keyframe kf;
			{
				std::unique_lock<std::mutex> lk(queue_mutex_);
				worker_busy_ = false; // previous keyframe fully processed
				idle_cv_.notify_all();
				queue_cv_.wait(lk, [this] { return !running_ || !kf_queue_.empty(); });
				if (!running_)
					break;
				kf = std::move(kf_queue_.front());
				kf_queue_.pop_front();
				worker_busy_ = true;
			}

			// 1. graph bookkeeping; seed T_opt from the previous OPTIMIZED pose so the
			// initial value stays consistent with the graph after large loop corrections
			if (kf.id == 0) {
				graph_.addPrior(
					0, toGtsam(kf.T_odom),
					gtsam::noiseModel::Diagonal::Sigmas(
						(gtsam::Vector6() << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3).finished())); // rot, then trans!
			} else {
				Sophus::SE3d rel = prevT_odom_.inverse() * kf.T_odom;
				graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(kf.id - 1, kf.id, toGtsam(rel), odom_noise_));
				kf.T_opt = prevT_opt_ * rel;
			}
			initial_.insert(kf.id, toGtsam(kf.T_opt));
			prevT_odom_ = kf.T_odom;
			{
				std::lock_guard<std::mutex> lk(kf_mutex_);
				keyframes_.push_back(kf);
			}

			// 2. loop candidates: radius search over OPTIMIZED poses, excluding recent.
			// Wide radius so the first true revisit is found even under large drift; the
			// multi-start margin test in verifyWithICP rejects wrong-lane matches, so try
			// several spatially distinct candidates — the nearest is often the wrong lane.
			// 3. verify with ICP, add loop edge
			bool loop_added = false;
			gtsam::Pose3 rel_out;
			double fitness = 0.0;
			int best = -1;
			for (int cand : findCandidates(kf, /*radius=*/10.0, /*exclude_last=*/30)) {
				if (verifyWithICP(kf, cand, rel_out, fitness)) {
					best = cand;
					break;
				}
			}
			if (best >= 0) {
				auto robust =
					gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1.0), loop_noise_);
				graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(best, kf.id, rel_out, robust));
				loop_added = true;
				RCLCPP_INFO(get_logger(), "Loop closure %d -> %d (rmse %.3f m)", kf.id, best, fitness);
			}

			// 4. optimize incrementally; refresh T_opt for all keyframes
			isam_.update(graph_, initial_);
			if (loop_added)
				for (int i = 0; i < 3; i++)
					isam_.update(); // extra relinearization after a loop
			graph_.resize(0);
			initial_.clear();
			gtsam::Values est = isam_.calculateEstimate();
			{
				std::lock_guard<std::mutex> lk(kf_mutex_);
				for (auto &k : keyframes_)
					k.T_opt = fromGtsam(est.at<gtsam::Pose3>(k.id));
				prevT_opt_ = keyframes_.back().T_opt;
			}
			{
				std::lock_guard<std::mutex> lk(tf_mutex_);
				T_map_odom_ = prevT_opt_ * prevT_odom_.inverse();
			}
			publishOptimizedPath(kf.stamp);
		}
	}

	// Block until every queued keyframe has been processed by the worker. save_map calls this so the
	// saved graph never ends before the odometry does
	void waitForQueueDrain() {
		std::unique_lock<std::mutex> lk(queue_mutex_);
		auto pending = [this] { return kf_queue_.size() + (worker_busy_ ? 1 : 0); };
		while (running_ && pending() > 0) {
			RCLCPP_INFO(get_logger(), "save_map: waiting for %zu queued keyframes", pending());
			idle_cv_.wait_for(lk, std::chrono::seconds(5), [&] { return !running_ || pending() == 0; });
		}
	}

	// Up to 3 candidates within radius, nearest first, forced >10 ids apart so the
	// alternatives are genuinely different places rather than neighbours of the nearest.
	std::vector<int> findCandidates(const Keyframe &kf, double radius, int exclude_last) {
		std::lock_guard<std::mutex> lk(kf_mutex_);
		std::vector<std::pair<double, int>> within;
		for (const auto &k : keyframes_) {
			if (k.id + exclude_last > kf.id)
				continue;
			double dist = (k.T_opt.translation() - kf.T_opt.translation()).norm();
			if (dist < radius)
				within.emplace_back(dist, k.id);
		}
		std::sort(within.begin(), within.end());
		std::vector<int> out;
		for (const auto &[dist, id] : within) {
			bool duplicate = false;
			for (int o : out)
				if (std::abs(o - id) <= 10)
					duplicate = true;
			if (!duplicate)
				out.push_back(id);
			if (out.size() == 3)
				break;
		}
		return out;
	}

	bool verifyWithICP(const Keyframe &kf, int cand_id, gtsam::Pose3 &rel_out, double &fitness) {
		// world-frame submap around the candidate keyframe
		KDCloud target;
		Sophus::SE3d T_cand;
		{
			std::lock_guard<std::mutex> lk(kf_mutex_);
			for (const auto &k : keyframes_) {
				if (std::abs(k.id - cand_id) > 5)
					continue;
				for (const auto &p : k.cloud_body)
					target.pts.push_back(k.T_opt * p);
			}
			T_cand = keyframes_[cand_id].T_opt; // ids are sequential, so index == id
		}
		if (target.pts.size() < 100)
			return false;
		KDTree tree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		tree.buildIndex();

		// Multi-start search with an ambiguity margin. ICP's capture range (~3 m) is smaller
		// than the drift can be at the first revisit, so run coarse ICP from the nominal
		// guess AND from starts on rings covering the candidate radius, clustering the
		// converged fits into distinct basins. Accept only when the best basin clearly beats
		// every other one — repeated structure (colonnades) and wrong-lane matches produce
		// near-equal competing basins and must be rejected, not fitted.
		struct Basin {
			Sophus::SE3d T;
			double rmse;
		};
		std::vector<Basin> basins;
		auto tryStart = [&](const Eigen::Vector3d &offset) {
			Sophus::SE3d T = Sophus::SE3d(kf.T_opt.so3(), kf.T_opt.translation() + offset);
			double rmse, inl;
			if (!alignICP(kf.cloud_body, target, tree, T, rmse, inl, /*max_pts=*/500) || inl < 0.75)
				return;
			for (auto &b : basins)
				if ((b.T.translation() - T.translation()).norm() < 0.3) {
					if (rmse < b.rmse)
						b = {T, rmse};
					return;
				}
			basins.push_back({T, rmse});
		};
		tryStart(Eigen::Vector3d::Zero());
		for (double ring : {3.0, 6.0})
			for (int k = 0; k < 8; ++k)
				tryStart(Eigen::Vector3d(ring * std::cos(k * M_PI / 4.0), ring * std::sin(k * M_PI / 4.0), 0));
		if (basins.empty())
			return false;
		std::sort(basins.begin(), basins.end(), [](const Basin &a, const Basin &b) { return a.rmse < b.rmse; });
		if (basins.size() > 1 && basins[1].rmse < 1.15 * basins[0].rmse) {
			RCLCPP_INFO(get_logger(), "Loop %d -> %d rejected: ambiguous (rmse %.3f vs %.3f, %.2f m apart)", kf.id,
						cand_id, basins[0].rmse, basins[1].rmse,
						(basins[1].T.translation() - basins[0].T.translation()).norm());
			return false;
		}

		// refine the winning basin at full resolution
		Sophus::SE3d T = basins[0].T;
		double inlier_ratio;
		if (!alignICP(kf.cloud_body, target, tree, T, fitness, inlier_ratio) || inlier_ratio < 0.75)
			return false;

		rel_out = toGtsam(T_cand.inverse() * T);
		return true;
	}

	// map -> odom: the drift correction that maps lio_node's odometry frame onto the
	// loop-closed global frame. Together with lio_node's odom -> base_link this gives
	// TF consumers a live, globally corrected pose without touching the smooth odometry.
	void publishMapToOdom(double stamp) {
		Sophus::SE3d T_map_odom;
		{
			std::lock_guard<std::mutex> lk(tf_mutex_);
			T_map_odom = T_map_odom_;
		}
		geometry_msgs::msg::TransformStamped tf;
		// Future-dated by a tolerance (AMCL-style) so exact-time lookups between our
		// publications never hit TF's extrapolation limit; the correction drifts slowly,
		// so being up to 0.5 s stale is negligible.
		tf.header.stamp = rclcpp::Time(static_cast<int64_t>((stamp + 0.5) * 1e9));
		tf.header.frame_id = "map";
		tf.child_frame_id = "odom";
		const Eigen::Vector3d &t = T_map_odom.translation();
		const Eigen::Quaterniond q = T_map_odom.unit_quaternion();
		tf.transform.translation.x = t.x();
		tf.transform.translation.y = t.y();
		tf.transform.translation.z = t.z();
		tf.transform.rotation.x = q.x();
		tf.transform.rotation.y = q.y();
		tf.transform.rotation.z = q.z();
		tf.transform.rotation.w = q.w();
		tf_broadcaster_->sendTransform(tf);
	}

	void publishOptimizedPath(double stamp) {
		if (path_pub_->get_subscription_count() == 0)
			return;
		nav_msgs::msg::Path path;
		path.header.stamp = rclcpp::Time(static_cast<int64_t>(stamp * 1e9));
		path.header.frame_id = "map"; // optimized poses live in the loop-closed frame
		{
			std::lock_guard<std::mutex> lk(kf_mutex_);
			path.poses.reserve(keyframes_.size());
			for (const auto &k : keyframes_) {
				geometry_msgs::msg::PoseStamped ps;
				ps.header = path.header;
				const Eigen::Vector3d &t = k.T_opt.translation();
				Eigen::Quaterniond q = k.T_opt.unit_quaternion();
				ps.pose.position.x = t.x();
				ps.pose.position.y = t.y();
				ps.pose.position.z = t.z();
				ps.pose.orientation.x = q.x();
				ps.pose.orientation.y = q.y();
				ps.pose.orientation.z = q.z();
				ps.pose.orientation.w = q.w();
				path.poses.push_back(ps);
			}
		}
		path_pub_->publish(path);
	}

	using SyncPolicy =
		message_filters::sync_policies::ExactTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;
	using Sync = message_filters::Synchronizer<SyncPolicy>;
	// Subcriber
	message_filters::Subscriber<nav_msgs::msg::Odometry> pose_sub_;
	message_filters::Subscriber<sensor_msgs::msg::PointCloud2> scan_sub_;
	std::shared_ptr<Sync> sync_;

	// Publisher / service
	rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
	std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
	rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;

	// members
	std::vector<Keyframe> keyframes_; // index == id; worker + service read/write → guard with kf_mutex_
	std::mutex kf_mutex_;
	std::deque<Keyframe> kf_queue_; // callback → worker handoff
	std::mutex queue_mutex_;
	std::condition_variable queue_cv_;
	std::condition_variable idle_cv_; // worker -> save_map: queue drained
	bool worker_busy_ = false;        // guarded by queue_mutex_
	std::thread loop_thread_;
	std::atomic<bool> running_{true};
	int next_id_ = 0; // guarded by queue_mutex_
	gtsam::NonlinearFactorGraph graph_;
	gtsam::Values initial_;
	gtsam::ISAM2 isam_{makeIsamParams()}; // incremental optimizer
	gtsam::SharedNoiseModel odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
		(gtsam::Vector6() << 1e-2, 1e-2, 1e-2, 5e-2, 5e-2, 5e-2).finished()); // rot, then trans!
	gtsam::SharedNoiseModel loop_noise_ =
		gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector6() << 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1).finished());
	Sophus::SE3d prevT_odom_, prevT_opt_; // worker thread only
	Sophus::SE3d last_kf_pose_;			  // callback thread only
	bool has_kf_ = false;
	Sophus::SE3d T_map_odom_; // worker writes, callback reads -> guard with tf_mutex_
	std::mutex tf_mutex_;
};

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<SlamNode>());
	rclcpp::shutdown();
	return 0;
}
