// Port of collision_avoidance/fcw/closing.py.
#include "fcw/closing.hpp"

#include <algorithm>
#include <cmath>

namespace fcw {

double median(std::vector<double> v) {
  const size_t n = v.size(), mid = n / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  const double upper = v[mid];
  if (n % 2) return upper;
  return 0.5 * (upper + *std::max_element(v.begin(), v.begin() + mid));
}

std::optional<double> roadConstant(const cv::Mat& flow, cv::Point2d foe, const cv::Matx33d& R, const cv::Matx33d& K,
                                   const std::vector<Box>& exclude, const ClosingConfig& cfg) {
  const int h = flow.rows, w = flow.cols;
  const double yh = foe.y;
  const int y1 = static_cast<int>(yh + cfg.band.first * (h - yh)), y2 = static_cast<int>(yh + cfg.band.second * (h - yh));
  if (y2 - y1 < 2 || yh >= h) return std::nullopt;
  const cv::Matx33d H = K * R * K.inv();
  std::vector<double> values;
  int n_points = 0;
  for (int y = std::max(y1, 0); y < std::min(y2, h); y += cfg.grid_step_y) {
    for (int x = 0; x < w; x += cfg.grid_step_x) {
      bool keep = true;
      for (const Box& b : exclude)
        if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) { keep = false; break; }
      if (!keep) continue;
      ++n_points;
      const cv::Vec3d r = H * cv::Vec3d(x, y, 1.0);
      const cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
      const double tx = f[0] - (r[0] / r[2] - x), ty = f[1] - (r[1] / r[2] - y);  // translational flow
      const double rx = x - foe.x, ry = y - foe.y, dist = std::hypot(rx, ry);
      const double outward = (tx * rx + ty * ry) / std::max(dist, 1e-9);
      if (dist > cfg.min_radius_px && outward > cfg.min_outward_px) values.push_back(dist / outward * (y - yh));
    }
  }
  if (n_points < cfg.min_points || static_cast<int>(values.size()) < cfg.min_points) return std::nullopt;
  return median(values);
}

std::optional<double> staticTtc(double c, const Box& box, cv::Point2d foe, const ClosingConfig& cfg) {
  const double contact = box.y2 - foe.y;
  return contact >= cfg.min_contact_px ? std::optional<double>(c / contact) : std::nullopt;
}

std::optional<double> ClosingTracker::updateC(std::optional<double> c) {
  if (c && std::isfinite(*c) && *c > 0) {
    c_history_.push_back(*c);
    if (static_cast<int>(c_history_.size()) > cfg_.history) c_history_.pop_front();
  }
  if (c_history_.empty()) return std::nullopt;
  return median({c_history_.begin(), c_history_.end()});
}

std::optional<double> ClosingTracker::update(int track_id, std::optional<double> kappa) {
  auto& hist = samples_[track_id];
  if (kappa && std::isfinite(*kappa)) {
    hist.push_back(*kappa);
    if (static_cast<int>(hist.size()) > cfg_.history) hist.pop_front();
  }
  if (hist.empty()) return std::nullopt;
  return median({hist.begin(), hist.end()});
}

void ClosingTracker::retain(const std::vector<int>& ids) {
  for (auto it = samples_.begin(); it != samples_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? samples_.erase(it) : std::next(it);
}

}  // namespace fcw
