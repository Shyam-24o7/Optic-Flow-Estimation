// Port of collision_avoidance/fcw/pipeline.py.
#include "fcw/pipeline.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fcw {

namespace {
double msSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

Level FcwFrame::level() const {
  Level out = Level::None;
  for (const auto& o : objects) out = std::max(out, o.level);
  return out;
}

const ObjectResult* FcwFrame::threat() const {
  const ObjectResult* best = nullptr;
  for (const auto& o : objects) {
    if (!o.course || !o.course->on_course || !o.estimate || !o.estimate->ttc_s) continue;
    if (!best || *o.estimate->ttc_s < *best->estimate->ttc_s) best = &o;
  }
  return best;
}

FlowSource disFlow() {
  auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
  return [dis](const cv::Mat& prev, const cv::Mat& curr) {
    cv::Mat flow;
    dis->calc(prev, curr, flow);
    return flow;
  };
}

FcwPipeline::FcwPipeline(FcwConfig cfg, DetectorFn detector, FlowSource flow)
    : cfg_(std::move(cfg)),
      detector_(std::move(detector)),
      flow_(std::move(flow)),
      K_(cfg_.K()),
      pp_(K_(0, 2), K_(1, 2)),
      tracker_(cfg_.tracker),
      ego_(K_, cfg_.ego),
      fusion_(cfg_.fusion),
      course_(cfg_.course) {}

FcwFrame FcwPipeline::process(const cv::Mat& frame_bgr, double t) {
  const auto t0 = std::chrono::steady_clock::now();
  FcwFrame out;
  ++index_;
  double dt = prev_t_ ? t - *prev_t_ : 0.0;
  if (dt <= 0) dt = 1.0 / cfg_.default_fps;  // first frame, or a repeated/out-of-order timestamp
  if (!prev_t_ || t > *prev_t_) prev_t_ = t;  // never move the clock backwards
  cv::resize(frame_bgr, out.frame, cv::Size(cfg_.frame_width, cfg_.frame_height));
  cv::Mat gray;
  cv::cvtColor(out.frame, gray, cv::COLOR_BGR2GRAY);
  const cv::Mat prev_gray = ring_.empty() ? cv::Mat() : std::get<2>(ring_.back());

  auto mark = std::chrono::steady_clock::now();
  out.flow = prev_gray.empty() ? cv::Mat::zeros(gray.size(), CV_32FC2) : flow_(prev_gray, gray);
  out.timings_ms["flow"] = msSince(mark);

  mark = std::chrono::steady_clock::now();
  std::vector<Detection> detections;
  for (const auto& d : detector_(out.frame))
    if (cfg_.course.class_width_m.count(d.cls)) detections.push_back(d);
  out.timings_ms["detection"] = msSince(mark);

  mark = std::chrono::steady_clock::now();
  std::vector<Box> prev_boxes;
  for (const auto& [id, tr] : tracker_.tracks()) prev_boxes.push_back(tr.bbox());
  const auto& tracks = tracker_.update(detections, dt, index_);
  out.ego = prev_gray.empty() ? EgoMotion{} : ego_.update(out.flow, prev_boxes);
  out.heading = out.ego.foe ? *out.ego.foe : pp_;
  const double yaw_rate = out.ego.rvec[1] / dt;
  ring_.emplace_back(index_, t, gray);
  if (static_cast<int>(ring_.size()) > kHistory) ring_.pop_front();
  out.timings_ms["track_ego"] = msSince(mark);

  mark = std::chrono::steady_clock::now();
  for (const ScaleTrack* tr : select(tracks))
    out.objects.push_back(assess(*tr, gray, prev_gray, out.flow, out.ego, out.heading, yaw_rate, t, dt));
  std::vector<int> live;
  for (const auto& [id, tr] : tracks) live.push_back(id);
  fusion_.retain(live);
  course_.retain(live);
  for (auto it = fsms_.begin(); it != fsms_.end();) it = tracks.count(it->first) ? std::next(it) : fsms_.erase(it);
  on_course_.clear();
  for (const auto& o : out.objects)
    if (o.course && o.course->on_course) on_course_.insert(o.track_id);
  out.timings_ms["ttc"] = msSince(mark);
  out.timings_ms["total"] = msSince(t0);

  out.index = index_;
  out.time_s = t;
  out.dt_s = dt;
  return out;
}

std::vector<const ScaleTrack*> FcwPipeline::select(const std::map<int, ScaleTrack>& tracks) const {
  // Last frame's on-course tracks first, then the largest boxes (stable, like Python's sorted).
  std::vector<const ScaleTrack*> order;
  for (const auto& [id, tr] : tracks) order.push_back(&tr);
  std::stable_sort(order.begin(), order.end(), [&](const ScaleTrack* a, const ScaleTrack* b) {
    const bool ca = !on_course_.count(a->id), cb = !on_course_.count(b->id);
    if (ca != cb) return ca < cb;
    const Box ba = a->bbox(), bb = b->bbox();
    return -(ba.width() * ba.height()) < -(bb.width() * bb.height());
  });
  if (static_cast<int>(order.size()) > cfg_.max_ttc_tracks) order.resize(cfg_.max_ttc_tracks);
  return order;
}

std::optional<std::pair<double, cv::Mat>> FcwPipeline::frameAt(int frame_index) const {
  for (const auto& [index, time_s, gray] : ring_)
    if (index == frame_index) return std::make_pair(time_s, gray);
  return std::nullopt;
}

ObjectResult FcwPipeline::assess(const ScaleTrack& track, const cv::Mat& gray, const cv::Mat& prev_gray, const cv::Mat& flow,
                                 const EgoMotion& ego, cv::Point2d heading, double yaw_rate, double t, double dt) {
  ObjectResult res;
  res.track_id = track.id;
  res.bbox = track.bbox();
  res.cls = track.cls;
  const Box& box = res.bbox;
  const EtaFilter* prior = fusion_.filter(track.id);
  const std::optional<double> eta_prior = prior ? prior->eta : std::nullopt;
  if (auto m = looming(track)) res.measurements.push_back(*m);
  if (!prev_gray.empty() && !track.lost) {
    const int k = chooseGap(eta_prior, dt, static_cast<int>(ring_.size()) - 1);
    const auto past = frameAt(index_ - k);
    const auto box_tk = track.boxAt(index_ - k);
    if (k && past && box_tk && t > past->first) {
      // The real span, not k * dt: frames can arrive irregularly.
      if (auto m = scaleTtc(gray, past->second, box, *box_tk, k, (t - past->first) / k, cfg_.scale)) res.measurements.push_back(*m);
    }
    const int level = chooseLevel(eta_prior, box, std::hypot(track.x[4], track.x[5]), dt);
    if (auto m = hornAtLevel(prev_gray, gray, box, pp_, level, dt, cfg_.horn)) res.measurements.push_back(*m);
    const cv::Mat mask = robustMask(flow, box, pp_, cfg_.divergence);
    const double rot = rotationDivergence(ego.R, K_, {box.cx(), box.cy()});
    if (auto m = divergenceTtc(flowMoments(flow, box, pp_, mask, cfg_.divergence.shrink), box, rot, 1, dt, cfg_.divergence))
      res.measurements.push_back(*m);
  }
  res.estimate = fusion_.update(track.id, t, dt, res.measurements);
  const std::optional<double> ttc = res.estimate ? res.estimate->ttc_s : std::nullopt;
  res.course = course_.update(track.id, t, box, track.cls, heading.x, heading.y, ttc, yaw_rate);
  WarningFsm& fsm = fsms_.try_emplace(track.id, cfg_.warning).first->second;
  res.level = fsm.step(res.course->on_course, ttc, res.estimate ? res.estimate->sigma_ttc_s : std::nullopt,
                       res.estimate ? static_cast<int>(res.estimate->recent_methods.size()) : 0);
  return res;
}

}  // namespace fcw
