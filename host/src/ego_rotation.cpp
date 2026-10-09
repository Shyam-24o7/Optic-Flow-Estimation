// Port of collision_avoidance/fcw/ego_rotation.py.
#include "fcw/ego_rotation.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

namespace fcw {

std::pair<cv::Mat, cv::Mat> gridSamples(const cv::Mat& flow, const std::vector<Box>& exclude, const EgoRotationConfig& cfg) {
  const int h = flow.rows, w = flow.cols, s = cfg.grid_step_px;
  std::vector<cv::Vec2d> prev, curr;
  for (int y = s / 2; y < h - cfg.hood_rows; y += s) {
    for (int x = s / 2; x < w; x += s) {
      bool keep = true;
      for (const Box& b : exclude) {
        const double mx = cfg.box_margin * (b.x2 - b.x1), my = cfg.box_margin * (b.y2 - b.y1);
        if (x >= b.x1 - mx && x <= b.x2 + mx && y >= b.y1 - my && y <= b.y2 + my) { keep = false; break; }
      }
      if (!keep) continue;
      const cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
      prev.emplace_back(x, y);
      curr.emplace_back(x + static_cast<double>(f[0]), y + static_cast<double>(f[1]));
    }
  }
  return {cv::Mat(prev, true).reshape(1), cv::Mat(curr, true).reshape(1)};
}

static double angleOf(const cv::Matx33d& R) {
  cv::Vec3d r;
  cv::Rodrigues(R, r);
  return cv::norm(r);
}

static cv::Vec3d rvecOf(const cv::Matx33d& R) {
  cv::Vec3d r;
  cv::Rodrigues(R, r);
  return r;
}

// Nearest rotation to K^-1 H K and the largest |singular value - 1|.
static std::pair<cv::Matx33d, double> rotationFromHomography(const cv::Mat& H, const cv::Matx33d& K) {
  cv::Mat A = cv::Mat(K.inv()) * H * cv::Mat(K);
  A /= std::cbrt(cv::determinant(A));
  cv::Mat w, u, vt;
  cv::SVD::compute(A, w, u, vt);
  cv::Mat R = u * vt;
  if (cv::determinant(R) < 0) R = -R;
  double deviation = 0;
  for (int i = 0; i < 3; ++i) deviation = std::max(deviation, std::abs(w.at<double>(i) - 1.0));
  return {cv::Matx33d(R), deviation};
}

std::optional<EgoMotion> estimate(const cv::Mat& prev, const cv::Mat& curr, const cv::Matx33d& K, const EgoRotationConfig& cfg) {
  const int n = prev.rows;
  if (n < cfg.min_points) return std::nullopt;
  cv::Mat e_mask;
  cv::Mat E = cv::findEssentialMat(prev, curr, cv::Mat(K), cv::RANSAC, cfg.ransac_confidence, cfg.ransac_threshold_px,
                                   cfg.max_iters, e_mask);
  const double e_ratio = E.empty() ? 0.0 : cv::countNonZero(e_mask) / static_cast<double>(n);
  // Pure rotation (or a stopped car) is degenerate for E: prefer the rotation homography
  // when it explains about as many points as E does.
  cv::Mat h_mask;
  cv::Mat H = cv::findHomography(prev, curr, cv::RANSAC, cfg.ransac_threshold_px, h_mask, cfg.max_iters);
  if (!H.empty()) {
    const auto [R_h, deviation] = rotationFromHomography(H, K);
    const double h_ratio = cv::countNonZero(h_mask) / static_cast<double>(n);
    if (deviation < cfg.rotation_only_tol && h_ratio >= cfg.rotation_only_share * e_ratio) {
      EgoMotion m;
      m.R = R_h;
      m.rvec = rvecOf(R_h);
      m.inlier_ratio = h_ratio;
      m.stationary = true;
      m.valid = true;
      return m;
    }
  }
  if (E.empty()) return std::nullopt;
  E = E.rowRange(0, 3);  // findEssentialMat may stack several solutions
  cv::Mat R, t;
  // recoverPose's own mask also drops far background, so E's RANSAC share is the confidence.
  std::vector<int> inliers;
  for (int i = 0; i < n; ++i)
    if (e_mask.at<uint8_t>(i)) inliers.push_back(i);
  if (cfg.pose_points > 0 && static_cast<int>(inliers.size()) > cfg.pose_points) {
    // Triangulating every inlier was half the frame time; a spread subset picks the same pose.
    cv::Mat p(cfg.pose_points, 2, CV_64F), c(cfg.pose_points, 2, CV_64F);
    const double step = static_cast<double>(inliers.size() - 1) / (cfg.pose_points - 1);
    for (int j = 0; j < cfg.pose_points; ++j) {
      const int i = inliers[static_cast<int>(j * step)];  // numpy linspace(...).astype(int)
      prev.row(i).copyTo(p.row(j));
      curr.row(i).copyTo(c.row(j));
    }
    cv::recoverPose(E, p, c, cv::Mat(K), R, t);
  } else {
    cv::Mat pose_mask = e_mask.clone();
    cv::recoverPose(E, prev, curr, cv::Mat(K), R, t, pose_mask);
  }
  EgoMotion m;
  m.R = cv::Matx33d(R);
  m.rvec = rvecOf(m.R);
  m.inlier_ratio = e_ratio;
  m.stationary = false;
  m.valid = true;
  cv::Vec3d tv(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  tv /= cv::norm(tv);
  if (std::abs(tv[2]) >= cfg.min_foe_tz) {
    const cv::Vec3d p = K * tv;
    m.foe = cv::Point2d(p[0] / p[2], p[1] / p[2]);
  }
  return m;
}

EgoMotion EgoRotationEstimator::hold() const {
  EgoMotion held = last_;
  held.inlier_ratio = 0;
  held.valid = false;
  return held;
}

EgoMotion EgoRotationEstimator::updatePoints(const cv::Mat& prev, const cv::Mat& curr) {
  std::optional<EgoMotion> est = estimate(prev, curr, K_, cfg_);
  if (!est || est->inlier_ratio < cfg_.min_inlier_ratio) return hold();
  if (last_.valid && std::abs(angleOf(est->R) - angleOf(last_.R)) > cfg_.max_jump_rad) return hold();
  if (est->foe && last_.foe) {
    const double a = cfg_.foe_smoothing;
    est->foe = cv::Point2d(a * est->foe->x + (1 - a) * last_.foe->x, a * est->foe->y + (1 - a) * last_.foe->y);
  }
  last_ = *est;
  return *est;
}

EgoMotion EgoRotationEstimator::update(const cv::Mat& flow, const std::vector<Box>& exclude) {
  const auto [prev, curr] = gridSamples(flow, exclude, cfg_);
  return updatePoints(prev, curr);
}

double rotationDivergence(const cv::Matx33d& R, const cv::Matx33d& K, cv::Point2d c) {
  const cv::Matx33d H = K * R * K.inv();
  auto mapped = [&](double x, double y) {
    const cv::Vec3d p = H * cv::Vec3d(x, y, 1.0);
    return cv::Point2d(p[0] / p[2], p[1] / p[2]);
  };
  const double d = 1.0;
  const double dgx = (mapped(c.x + d, c.y).x - mapped(c.x - d, c.y).x) / (2 * d);
  const double dgy = (mapped(c.x, c.y + d).y - mapped(c.x, c.y - d).y) / (2 * d);
  return dgx + dgy - 2.0;
}

}  // namespace fcw
