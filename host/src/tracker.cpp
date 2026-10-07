// Port of collision_avoidance/fcw/tracker.py and looming.py.
#include "fcw/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fcw {

cv::Vec4d boxToZ(const Box& b) {
  const double w = std::max(b.x2 - b.x1, 1.0), h = std::max(b.y2 - b.y1, 1.0);
  return {(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, 0.5 * std::log(w * h), w / h};
}

Box zToBox(const State& x) {
  const double root = std::exp(x[2]);
  const double w = root * std::sqrt(x[3]), h = root / std::sqrt(x[3]);
  return {x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2};
}

double iou(const Box& a, const Box& b) {
  const double xa = std::max(a.x1, b.x1), ya = std::max(a.y1, b.y1);
  const double xb = std::min(a.x2, b.x2), yb = std::min(a.y2, b.y2);
  const double inter = std::max(0.0, xb - xa) * std::max(0.0, yb - ya);
  const double area_a = (a.x2 - a.x1) * (a.y2 - a.y1), area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6);
}

std::optional<Box> ScaleTrack::boxAt(int frame_index) const {
  for (const auto& [index, box] : history)
    if (index == frame_index) return box;
  return std::nullopt;
}

// Hungarian algorithm (shortest augmenting path, O(n^2 m)) on an n x m matrix with n <= m.
static std::vector<int> hungarian(const std::vector<std::vector<double>>& a) {
  const int n = static_cast<int>(a.size()), m = static_cast<int>(a[0].size());
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1), v(m + 1);
  std::vector<int> p(m + 1), way(m + 1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, inf);
    std::vector<char> used(m + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  std::vector<int> row_to_col(n, -1);
  for (int j = 1; j <= m; ++j)
    if (p[j]) row_to_col[p[j] - 1] = j - 1;
  return row_to_col;
}

std::vector<std::pair<int, int>> linearSumAssignment(const std::vector<std::vector<double>>& cost) {
  std::vector<std::pair<int, int>> out;
  if (cost.empty() || cost[0].empty()) return out;
  const size_t n = cost.size(), m = cost[0].size();
  if (n <= m) {
    const auto rc = hungarian(cost);
    for (size_t r = 0; r < n; ++r) out.emplace_back(static_cast<int>(r), rc[r]);
  } else {
    std::vector<std::vector<double>> t(m, std::vector<double>(n));
    for (size_t r = 0; r < n; ++r)
      for (size_t c = 0; c < m; ++c) t[c][r] = cost[r][c];
    const auto cr = hungarian(t);
    for (size_t c = 0; c < m; ++c) out.emplace_back(cr[c], static_cast<int>(c));
    std::sort(out.begin(), out.end());
  }
  return out;
}

ScaleTracker::ScaleTracker(TrackerConfig cfg) : cfg_(cfg) {
  R_ = cv::Matx44d::diag({cfg.meas_center_px * cfg.meas_center_px, cfg.meas_center_px * cfg.meas_center_px,
                          cfg.meas_scale * cfg.meas_scale, cfg.meas_aspect * cfg.meas_aspect});
}

void ScaleTracker::predict(ScaleTrack& t, double dt) const {
  Cov F = Cov::eye();
  F(0, 4) = F(1, 5) = F(2, 6) = dt;
  Cov Q = Cov::zeros();
  const int pos[3] = {0, 1, 2}, vel[3] = {4, 5, 6};
  const double sigma[3] = {cfg_.accel_px_s2, cfg_.accel_px_s2, cfg_.scale_accel_s2};
  for (int k = 0; k < 3; ++k) {
    const double q = sigma[k] * sigma[k];
    Q(pos[k], pos[k]) = q * dt * dt * dt / 3;
    Q(pos[k], vel[k]) = Q(vel[k], pos[k]) = q * dt * dt / 2;
    Q(vel[k], vel[k]) = q * dt;
  }
  Q(3, 3) = cfg_.aspect_walk * cfg_.aspect_walk * dt;
  t.x = F * t.x;
  t.P = F * t.P * F.t() + Q;
}

void ScaleTracker::correct(ScaleTrack& t, const Detection& d) const {
  cv::Matx<double, 4, 7> H = cv::Matx<double, 4, 7>::zeros();
  for (int k = 0; k < 4; ++k) H(k, k) = 1;
  const cv::Vec4d z = boxToZ(d.box);
  const cv::Vec4d y = z - H * t.x;
  const cv::Matx44d S = H * t.P * H.t() + R_;
  const cv::Matx<double, 7, 4> K = t.P * H.t() * S.inv(cv::DECOMP_LU);
  t.x = t.x + K * y;
  t.P = (Cov::eye() - K * H) * t.P;
  t.cls = d.cls;
  t.lost = 0;
  t.age += 1;
}

void ScaleTracker::create(const Detection& d) {
  ScaleTrack t;
  t.id = next_id_;
  t.x = State::all(0);
  const cv::Vec4d z = boxToZ(d.box);
  for (int k = 0; k < 4; ++k) t.x[k] = z[k];
  t.P = Cov::diag({R_(0, 0), R_(1, 1), R_(2, 2), R_(3, 3), 100.0 * 100.0, 100.0 * 100.0, 1.0});
  t.cls = d.cls;
  tracks_[next_id_++] = t;
}

const std::map<int, ScaleTrack>& ScaleTracker::update(const std::vector<Detection>& dets, double dt, int frame_index) {
  for (auto& [id, t] : tracks_) predict(t, dt);

  std::vector<int> ids;
  for (const auto& [id, t] : tracks_) ids.push_back(id);
  std::vector<char> track_matched(ids.size(), false), det_matched(dets.size(), false);
  if (!ids.empty() && !dets.empty()) {
    std::vector<std::vector<double>> ious(ids.size(), std::vector<double>(dets.size()));
    std::vector<std::vector<double>> cost(ids.size(), std::vector<double>(dets.size()));
    for (size_t r = 0; r < ids.size(); ++r)
      for (size_t c = 0; c < dets.size(); ++c) {
        ious[r][c] = iou(tracks_.at(ids[r]).bbox(), dets[c].box);
        cost[r][c] = -ious[r][c];
      }
    for (const auto& [r, c] : linearSumAssignment(cost)) {
      if (ious[r][c] >= cfg_.iou_threshold) {
        correct(tracks_.at(ids[r]), dets[c]);
        track_matched[r] = det_matched[c] = true;
      }
    }
  }
  for (size_t r = 0; r < ids.size(); ++r)
    if (!track_matched[r]) tracks_.at(ids[r]).lost += 1;
  for (size_t c = 0; c < dets.size(); ++c)
    if (!det_matched[c]) create(dets[c]);
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    it = it->second.lost > cfg_.max_lost ? tracks_.erase(it) : std::next(it);
  }
  for (auto& [id, t] : tracks_) {
    if (t.lost) continue;  // predicted boxes must not stand in for measurements
    t.history.emplace_back(frame_index, t.bbox());
    if (static_cast<int>(t.history.size()) > kHistory) t.history.pop_front();
  }
  return tracks_;
}

std::optional<Measurement> looming(const ScaleTrack& t) {
  if (t.age < 3 || t.lost) return std::nullopt;
  return Measurement{"looming", t.scaleRate(), t.scaleRateVar()};
}

}  // namespace fcw
