// Port of collision_avoidance/fcw/divergence.py.
#include "fcw/divergence.hpp"

#include <cmath>
#include <random>

namespace fcw {

std::optional<FlowMoments> flowMoments(const cv::Mat& flow, const Box& b, cv::Point2d pp, const cv::Mat& mask, double shrink) {
  const int h = flow.rows, w = flow.cols;
  const double mx = shrink / 2 * (b.x2 - b.x1), my = shrink / 2 * (b.y2 - b.y1);
  const int x1 = std::max(0, static_cast<int>(std::ceil(b.x1 + mx))), y1 = std::max(0, static_cast<int>(std::ceil(b.y1 + my)));
  const int x2 = std::min(w, static_cast<int>(std::floor(b.x2 - mx))), y2 = std::min(h, static_cast<int>(std::floor(b.y2 - my)));
  if (x2 <= x1 || y2 <= y1) return std::nullopt;
  FlowMoments m{};
  for (int yy = y1; yy < y2; ++yy) {
    for (int xx = x1; xx < x2; ++xx) {
      if (!mask.empty() && !mask.at<uint8_t>(yy, xx)) continue;
      const double x = xx - pp.x, y = yy - pp.y;
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      const double u = f[0], v = f[1];
      m[0] += 1; m[1] += x; m[2] += y; m[3] += x * x; m[4] += x * y; m[5] += y * y;
      m[6] += u; m[7] += v; m[8] += x * u; m[9] += y * u; m[10] += x * v; m[11] += y * v;
    }
  }
  return m;
}

std::optional<std::pair<cv::Vec3d, cv::Vec3d>> affineFromMoments(const FlowMoments& m) {
  const cv::Matx33d A(m[0], m[1], m[2], m[1], m[3], m[4], m[2], m[4], m[5]);
  if (m[0] < 3 || std::abs(cv::determinant(A)) < 1e-9) return std::nullopt;
  cv::Vec3d a, b;
  cv::solve(A, cv::Vec3d(m[6], m[8], m[9]), a, cv::DECOMP_LU);
  cv::solve(A, cv::Vec3d(m[7], m[10], m[11]), b, cv::DECOMP_LU);
  return std::make_pair(a, b);
}

std::optional<Measurement> divergenceTtc(const std::optional<FlowMoments>& m, const Box& b, double rotation_div, int k,
                                         double dt, const DivergenceConfig& cfg) {
  if (!m || (*m)[0] < cfg.min_pixels) return std::nullopt;
  const auto fit = affineFromMoments(*m);
  if (!fit) return std::nullopt;
  const double div = fit->first[1] + fit->second[2] - rotation_div;
  const double w = b.x2 - b.x1, h = b.y2 - b.y1;
  const double var_div = cfg.flow_noise_px * cfg.flow_noise_px * 12.0 / (*m)[0] * (1.0 / (w * w) + 1.0 / (h * h));
  const double span = 2.0 * k * dt;
  return Measurement{"divergence", div / span, var_div / (span * span)};
}

namespace {

struct Region { int x1, y1, x2, y2; };

std::optional<Region> robustRegion(const cv::Mat& flow, const Box& b) {
  const Region r{std::max(0, static_cast<int>(b.x1)), std::max(0, static_cast<int>(b.y1)),
                 std::min(flow.cols, static_cast<int>(std::ceil(b.x2))), std::min(flow.rows, static_cast<int>(std::ceil(b.y2)))};
  if (r.x2 - r.x1 < 3 || r.y2 - r.y1 < 3) return std::nullopt;
  return r;
}

cv::Mat ransac(const cv::Mat& flow, const Region& r, cv::Point2d pp, const DivergenceConfig& cfg,
               const std::vector<std::array<int, 3>>& samples) {
  cv::Mat mask = cv::Mat::ones(flow.size(), CV_8U);
  std::vector<cv::Vec3d> X;
  std::vector<double> U, V;
  for (int yy = r.y1; yy < r.y2; yy += cfg.ransac_stride)
    for (int xx = r.x1; xx < r.x2; xx += cfg.ransac_stride) {
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      X.emplace_back(1.0, xx - pp.x, yy - pp.y);
      U.push_back(f[0]);
      V.push_back(f[1]);
    }
  bool found = false;
  cv::Vec3d best_a, best_b;
  long best_count = -1;
  for (const auto& idx : samples) {
    const cv::Matx33d A(X[idx[0]][0], X[idx[0]][1], X[idx[0]][2], X[idx[1]][0], X[idx[1]][1], X[idx[1]][2],
                        X[idx[2]][0], X[idx[2]][1], X[idx[2]][2]);
    cv::Vec3d a, b;
    if (!cv::solve(A, cv::Vec3d(U[idx[0]], U[idx[1]], U[idx[2]]), a, cv::DECOMP_LU)) continue;
    if (!cv::solve(A, cv::Vec3d(V[idx[0]], V[idx[1]], V[idx[2]]), b, cv::DECOMP_LU)) continue;
    long count = 0;
    for (size_t i = 0; i < X.size(); ++i)
      if (std::hypot(X[i].dot(a) - U[i], X[i].dot(b) - V[i]) < cfg.inlier_px) ++count;
    if (count > best_count) { best_count = count; best_a = a; best_b = b; found = true; }
  }
  if (!found) return mask;
  for (int yy = r.y1; yy < r.y2; ++yy)
    for (int xx = r.x1; xx < r.x2; ++xx) {
      const double x = xx - pp.x, y = yy - pp.y;
      const cv::Vec2f f = flow.at<cv::Vec2f>(yy, xx);
      const double res = std::hypot(f[0] - (best_a[0] + best_a[1] * x + best_a[2] * y), f[1] - (best_b[0] + best_b[1] * x + best_b[2] * y));
      mask.at<uint8_t>(yy, xx) = res < cfg.inlier_px ? 1 : 0;
    }
  return mask;
}

}  // namespace

cv::Mat robustMask(const cv::Mat& flow, const Box& b, cv::Point2d pp, const DivergenceConfig& cfg,
                   const std::vector<std::array<int, 3>>& samples) {
  const auto r = robustRegion(flow, b);
  return r ? ransac(flow, *r, pp, cfg, samples) : cv::Mat::ones(flow.size(), CV_8U);
}

cv::Mat robustMask(const cv::Mat& flow, const Box& b, cv::Point2d pp, const DivergenceConfig& cfg) {
  const auto r = robustRegion(flow, b);
  if (!r) return cv::Mat::ones(flow.size(), CV_8U);
  const int nx = (r->x2 - r->x1 + cfg.ransac_stride - 1) / cfg.ransac_stride;
  const int ny = (r->y2 - r->y1 + cfg.ransac_stride - 1) / cfg.ransac_stride;
  const int n = nx * ny;
  std::mt19937 rng(0);
  std::vector<std::array<int, 3>> samples;
  for (int it = 0; it < cfg.ransac_iterations; ++it) {
    std::array<int, 3> s{};
    for (int j = 0; j < 3; ++j) {
      bool unique;
      do {
        s[j] = std::uniform_int_distribution<int>(0, n - 1)(rng);
        unique = true;
        for (int q = 0; q < j; ++q) unique &= s[q] != s[j];
      } while (!unique);
    }
    samples.push_back(s);
  }
  return ransac(flow, *r, pp, cfg, samples);
}

}  // namespace fcw
