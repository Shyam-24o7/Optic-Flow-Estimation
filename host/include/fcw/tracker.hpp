// Log-scale Kalman tracker. Port of collision_avoidance/fcw/tracker.py and looming.py.
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "fcw/types.hpp"

namespace fcw {

constexpr int kHistory = 9;  // boxes kept per track: current + 8 past frames

struct TrackerConfig {
  double iou_threshold = 0.3;
  int max_lost = 30;
  double accel_px_s2 = 200.0;
  double scale_accel_s2 = 1.0;
  double aspect_walk = 0.1;
  double meas_center_px = 2.0;
  double meas_scale = 0.02;
  double meas_aspect = 0.05;
};

using State = cv::Vec<double, 7>;     // [cx, cy, s = ln sqrt(w h), a = w/h, vx, vy, vs]
using Cov = cv::Matx<double, 7, 7>;

cv::Vec4d boxToZ(const Box& b);
Box zToBox(const State& x);
double iou(const Box& a, const Box& b);  // same formula as collision_avoidance/tracking.py

struct ScaleTrack {
  int id = 0;
  State x;
  Cov P;
  std::string cls;
  int lost = 0;
  int age = 1;
  std::deque<std::pair<int, Box>> history;  // (frame index, box), detected frames only

  Box bbox() const { return zToBox(x); }
  double scaleRate() const { return x[6]; }
  double scaleRateVar() const { return P(6, 6); }
  std::optional<Box> boxAt(int frame_index) const;
};

class ScaleTracker {
 public:
  explicit ScaleTracker(TrackerConfig cfg = {});
  const std::map<int, ScaleTrack>& update(const std::vector<Detection>& detections, double dt, int frame_index);
  const std::map<int, ScaleTrack>& tracks() const { return tracks_; }

 private:
  void predict(ScaleTrack& t, double dt) const;
  void correct(ScaleTrack& t, const Detection& d) const;
  void create(const Detection& d);

  TrackerConfig cfg_;
  std::map<int, ScaleTrack> tracks_;
  int next_id_ = 0;
  cv::Matx44d R_;
};

// Rectangular linear assignment minimising cost (scipy.optimize.linear_sum_assignment semantics).
std::vector<std::pair<int, int>> linearSumAssignment(const std::vector<std::vector<double>>& cost);

std::optional<Measurement> looming(const ScaleTrack& t);  // None until 3 detections and while lost

}  // namespace fcw
