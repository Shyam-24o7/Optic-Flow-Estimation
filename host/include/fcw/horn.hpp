// Horn direct time-to-contact. Port of collision_avoidance/fcw/horn.py.
//
// Integer definition (the HLS engine of plan 3 must match it bit for bit):
//   S = prev + curr; Ex, Ey = 3x3 Sobel of S on interior pixels (12 bit); Et = curr - prev (9 bit);
//   G = x*Ex + y*Ey with x, y measured from the rounded principal point.
// Ten int64 sums per box: Ex^2, ExEy, ExG, Ey^2, EyG, G^2, ExEt, EyEt, GEt, Et^2. C = 16 * C'.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <optional>

#include "fcw/types.hpp"

namespace fcw {

constexpr const char* kHornTerms[10] = {"ExEx", "ExEy", "ExG", "EyEy", "EyG", "GG", "ExEt", "EyEt", "GEt", "EtEt"};
constexpr int kSobelGain = 16;

struct HornConfig {
  int grad_threshold_l1 = 64;
  double shrink = 0.10;
  double max_condition = 1e4;
  double max_rel_residual = 0.5;
  int64_t min_pixels = 100;
};

struct HornSums {
  std::array<int64_t, 10> s{};
  int64_t n = 0;
};

// Pixel range [x1, x2) x [y1, y2) of the shrunk box, clipped to the Sobel interior.
std::optional<cv::Rect> innerBox(const Box& box, double shrink, int width, int height);
std::optional<HornSums> hornSums(const cv::Mat& prev, const cv::Mat& curr, const Box& box, cv::Point2d principal,
                                 const HornConfig& cfg = {});
std::optional<Measurement> hornSolve(const HornSums& sums, double dt, const HornConfig& cfg = {});
int chooseLevel(std::optional<double> eta, const Box& box, double center_speed_px_s, double dt, int levels = 3);
cv::Mat downsample(const cv::Mat& gray, int level);
std::optional<Measurement> hornAtLevel(const cv::Mat& prev, const cv::Mat& curr, const Box& box, cv::Point2d principal,
                                       int level, double dt, const HornConfig& cfg = {});

}  // namespace fcw
