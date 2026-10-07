// Port of collision_avoidance/fcw/scale_search.py.
#include "fcw/scale_search.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace fcw {

cv::Mat crop(const cv::Mat& gray, cv::Point2d c, cv::Size2d size, int patch) {
  const double sx = size.width / patch, sy = size.height / patch;
  // patch pixel i maps to image point c - size/2 + (i + 0.5) * s, in OpenCV's pixel-centre convention.
  const cv::Matx23d M(sx, 0, c.x - size.width / 2 + 0.5 * sx - 0.5, 0, sy, c.y - size.height / 2 + 0.5 * sy - 0.5);
  cv::Mat out;
  cv::warpAffine(gray, out, M, cv::Size(patch, patch), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
  out.convertTo(out, CV_32F);
  return out;
}

double ncc(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat da, db;
  a.convertTo(da, CV_64F);
  b.convertTo(db, CV_64F);
  da -= cv::mean(da)[0];
  db -= cv::mean(db)[0];
  const double denom = std::sqrt(da.dot(da) * db.dot(db));
  return denom > 0 ? da.dot(db) / denom : 0.0;
}

static double parabolaPeak(double cm, double c0, double cp) {
  const double denom = cm - 2 * c0 + cp;
  return denom >= 0 ? 0.0 : std::clamp(0.5 * (cm - cp) / denom, -0.5, 0.5);
}

std::optional<std::pair<double, double>> searchScale(const cv::Mat& gray_t, const cv::Mat& gray_tk, const Box& bt,
                                                     const Box& btk, const ScaleSearchConfig& cfg) {
  const double w = bt.width() * (1 - cfg.shrink), h = bt.height() * (1 - cfg.shrink);
  if (std::min(w, h) < cfg.min_box_px) return std::nullopt;
  const cv::Point2d ct(bt.cx(), bt.cy()), ctk(btk.cx(), btk.cy());
  const cv::Mat tmpl = crop(gray_t, ct, {w, h}, cfg.patch);
  auto score = [&](double s) { return ncc(tmpl, crop(gray_tk, ctk, {w / s, h / s}, cfg.patch)); };

  // numpy.arange(s_min, s_max + 1e-9, coarse_step)
  const int n_coarse = static_cast<int>(std::ceil((cfg.s_max + 1e-9 - cfg.s_min) / cfg.coarse_step));
  double best = cfg.s_min, best_score = -2;
  for (int i = 0; i < n_coarse; ++i) {
    const double s = cfg.s_min + i * cfg.coarse_step;
    const double sc = score(s);
    if (sc > best_score) { best_score = sc; best = s; }  // first maximum, like argmax
  }
  const int half = cfg.fine_count / 2;
  std::vector<double> fine, scores;
  for (int j = -half; j <= half; ++j) {
    fine.push_back(best + cfg.fine_step * j);
    scores.push_back(score(fine.back()));
  }
  const int i = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
  if (i == 0 || i == static_cast<int>(fine.size()) - 1) return std::nullopt;  // peak not bracketed
  const double s_star = fine[i] + cfg.fine_step * parabolaPeak(scores[i - 1], scores[i], scores[i + 1]);
  if (!(cfg.s_min < s_star && s_star < cfg.s_max)) return std::nullopt;
  return std::make_pair(s_star, scores[i]);
}

std::optional<Measurement> scaleTtc(const cv::Mat& gray_t, const cv::Mat& gray_tk, const Box& bt, const Box& btk, int k,
                                    double dt, const ScaleSearchConfig& cfg) {
  const auto found = searchScale(gray_t, gray_tk, bt, btk, cfg);
  if (!found || found->second < cfg.min_ncc) return std::nullopt;
  const auto [s, peak] = *found;
  const double span = k * dt;
  const double sigma_s = cfg.fine_step * (1.0 + 10.0 * (1.0 - peak));
  return Measurement{"scale", (s - 1.0) / span, (sigma_s / span) * (sigma_s / span)};
}

int chooseGap(std::optional<double> eta, double dt, int available, double min_expansion) {
  std::vector<int> gaps;
  for (int k : {1, 2, 4, 8})
    if (k <= available) gaps.push_back(k);
  if (gaps.empty()) return 0;
  if (!eta || *eta <= 0) return gaps.back();
  for (int k : gaps)
    if (*eta * k * dt >= min_expansion) return k;
  return gaps.back();
}

}  // namespace fcw
