// Flow divergence per box from an affine fit, after removing camera rotation.
// Port of collision_avoidance/fcw/divergence.py.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

// n, x, y, xx, xy, yy, u, v, xu, yu, xv, yv (x, y from the principal point).
using FlowMoments = std::array<double, 12>;

struct DivergenceConfig {
  double shrink = 0.10;
  double flow_noise_px = 0.3;
  double inlier_px = 1.0;
  int ransac_iterations = 64;
  int ransac_stride = 2;
  double min_pixels = 50;
};

// flow: CV_32FC2; mask: optional CV_8U / bool-like, same size, nonzero = use the pixel.
std::optional<FlowMoments> flowMoments(const cv::Mat& flow, const Box& box, cv::Point2d principal, const cv::Mat& mask = {},
                                       double shrink = 0.10);
std::optional<std::pair<cv::Vec3d, cv::Vec3d>> affineFromMoments(const FlowMoments& m);
std::optional<Measurement> divergenceTtc(const std::optional<FlowMoments>& m, const Box& box, double rotation_div, int k,
                                         double dt, const DivergenceConfig& cfg = {});
// Pixels that follow the dominant affine flow in the box (RANSAC). Returns CV_8U, 1 = keep.
cv::Mat robustMask(const cv::Mat& flow, const Box& box, cv::Point2d principal, const DivergenceConfig& cfg = {});
// Test hook: RANSAC with given sample triples (indices into the stride-subsampled box pixels).
cv::Mat robustMask(const cv::Mat& flow, const Box& box, cv::Point2d principal, const DivergenceConfig& cfg,
                   const std::vector<std::array<int, 3>>& samples);

}  // namespace fcw
