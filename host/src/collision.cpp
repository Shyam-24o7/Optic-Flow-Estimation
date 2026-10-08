// Port of collision_avoidance/fcw/collision.py.
#include "fcw/collision.hpp"

#include <algorithm>
#include <cmath>

namespace fcw {

CourseResult CourseChecker::update(int track_id, double t, const Box& b, const std::string& cls, double heading_x,
                                   double horizon_y, std::optional<double> ttc_s, double yaw_rate_rps) {
  CourseResult res;
  res.r = ((b.x1 + b.x2) / 2 - heading_x) / std::max(b.x2 - b.x1, 1.0);
  auto& hist = history_[track_id];
  hist.emplace_back(t, res.r);
  while (!hist.empty() && t - hist.front().first > cfg_.history_s) hist.pop_front();
  const auto width = cfg_.class_width_m.find(cls);
  if (width == cfg_.class_width_m.end()) return res;
  res.width_m = width->second;
  const double lateral = res.r * width->second;
  const double overlap = std::min(lateral + width->second / 2, cfg_.ego_width_m / 2) - std::max(lateral - width->second / 2, -cfg_.ego_width_m / 2);
  res.in_path = b.y2 >= horizon_y && overlap >= cfg_.min_overlap * std::min(width->second, cfg_.ego_width_m);
  res.threshold = 0.5 * (1.0 + (cfg_.ego_width_m + cfg_.margin_m) / width->second);
  if (std::abs(yaw_rate_rps) > cfg_.turn_yaw_rate_rps) res.threshold *= cfg_.turn_widen;
  if (b.y2 < horizon_y || !ttc_s) return res;
  double r_dot = 0;
  // Needs a real time span: live cameras repeat timestamps.
  if (hist.size() >= 3 && hist.back().first - hist.front().first > 1e-3) {
    double tm = 0, rm = 0;
    for (const auto& [ts, rs] : hist) { tm += ts; rm += rs; }
    tm /= hist.size();
    rm /= hist.size();
    double num = 0, den = 0;
    for (const auto& [ts, rs] : hist) { num += (ts - tm) * (rs - rm); den += (ts - tm) * (ts - tm); }
    r_dot = num / den;
  }
  res.r_contact = res.r + r_dot * *ttc_s;
  res.on_course = std::abs(*res.r_contact) < res.threshold;
  return res;
}

void CourseChecker::retain(const std::vector<int>& ids) {
  for (auto it = history_.begin(); it != history_.end();)
    it = std::find(ids.begin(), ids.end(), it->first) == ids.end() ? history_.erase(it) : std::next(it);
}

Level WarningFsm::step(bool on_course, std::optional<double> ttc_s, std::optional<double> sigma, int recent_methods) {
  const bool can_raise = on_course && ttc_s && recent_methods >= cfg_.min_methods;
  const bool critical = can_raise && *ttc_s + sigma.value_or(0.0) <= cfg_.critical_ttc_s;
  const bool warn = can_raise && *ttc_s <= cfg_.warn_ttc_s;
  if (level_ == Level::None) {
    warn_count_ = warn ? warn_count_ + 1 : 0;
    if (critical) enter(Level::Critical);
    else if (warn_count_ >= cfg_.warn_frames) enter(Level::Warning);
  } else if (level_ == Level::Warning) {
    if (critical) {
      enter(Level::Critical);
    } else {
      const bool clearing = !on_course || !ttc_s || *ttc_s > cfg_.warn_clear_ttc_s;
      clear_count_ = clearing ? clear_count_ + 1 : 0;
      if (clear_count_ >= cfg_.clear_frames) enter(Level::None);
    }
  } else {
    const bool clearing = !on_course || !ttc_s || *ttc_s > cfg_.critical_clear_ttc_s;
    clear_count_ = clearing ? clear_count_ + 1 : 0;
    if (clear_count_ >= cfg_.clear_frames) enter(Level::Warning);
  }
  return level_;
}

}  // namespace fcw
