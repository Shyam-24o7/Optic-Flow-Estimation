// Scale-ratio search between frames t-k and t. Port of collision_avoidance/fcw/scale_search.py.
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <utility>

#include "fcw/types.hpp"

namespace fcw {

struct ScaleSearchConfig {
  int patch = 32;
  double shrink = 0.10;
  double s_min = 0.95;
  double s_max = 1.25;
  double coarse_step = 0.02;
  double fine_step = 0.004;
  int fine_count = 9;
  double min_ncc = 0.5;
  double min_box_px = 12.0;
};

cv::Mat crop(const cv::Mat& gray, cv::Point2d center, cv::Size2d size, int patch);  // CV_32F patch x patch
double ncc(const cv::Mat& a, const cv::Mat& b);

// Best s = size(t) / size(t-k) and its NCC peak, or none.
std::optional<std::pair<double, double>> searchScale(const cv::Mat& gray_t, const cv::Mat& gray_tk, const Box& box_t,
                                                     const Box& box_tk, const ScaleSearchConfig& cfg = {});
std::optional<Measurement> scaleTtc(const cv::Mat& gray_t, const cv::Mat& gray_tk, const Box& box_t, const Box& box_tk,
                                    int k, double dt, const ScaleSearchConfig& cfg = {});
// Smallest gap in {1, 2, 4, 8} whose expected expansion reaches min_expansion (0 when no history).
int chooseGap(std::optional<double> eta, double dt, int available, double min_expansion = 0.02);

}  // namespace fcw
