// Parity: fcw::ScaleTracker vs collision_avoidance/fcw/tracker.py (golden file tracker.yml.gz).
#include <gtest/gtest.h>

#include <sstream>

#include "fcw/tracker.hpp"
#include "golden.hpp"

namespace {

std::vector<fcw::Detection> detections(const golden::File& g, const std::string& key) {
  cv::Mat boxes = g.mat(key + "_dets");
  std::vector<std::string> names;
  std::stringstream ss(g.str(key + "_cls"));
  for (std::string n; std::getline(ss, n, ',');) names.push_back(n);
  std::vector<fcw::Detection> out;
  for (int r = 0; r < boxes.rows; ++r) {
    out.push_back({{boxes.at<double>(r, 0), boxes.at<double>(r, 1), boxes.at<double>(r, 2), boxes.at<double>(r, 3)}, names.at(r), 0.9});
  }
  return out;
}

void replay(const std::string& name) {
  golden::File g("tracker");
  const double dt = g.num("dt");
  fcw::ScaleTracker tracker;
  const int frames = static_cast<int>(g.num(name + "_frames"));
  for (int i = 0; i < frames; ++i) {
    const std::string key = name + "_f" + std::to_string(i);
    const auto& tracks = tracker.update(detections(g, key), dt, i);
    cv::Mat ids = g.mat(key + "_ids");
    ASSERT_EQ(static_cast<int>(tracks.size()), ids.rows) << key;
    for (int r = 0; r < ids.rows; ++r) {
      const int id = static_cast<int>(ids.at<double>(r, 0));
      const std::string tk = key + "_t" + std::to_string(id);
      ASSERT_TRUE(tracks.count(id)) << tk;
      const fcw::ScaleTrack& t = tracks.at(id);
      cv::Mat x = g.mat(tk + "_x"), P = g.mat(tk + "_P");
      for (int k = 0; k < 7; ++k) EXPECT_NEAR(t.x[k], x.at<double>(k, 0), 1e-9) << tk << " x[" << k << "]";
      for (int a = 0; a < 7; ++a)
        for (int b = 0; b < 7; ++b) EXPECT_NEAR(t.P(a, b), P.at<double>(a, b), 1e-9 * (1 + std::abs(P.at<double>(a, b)))) << tk;
      EXPECT_EQ(t.lost, static_cast<int>(g.num(tk + "_lost"))) << tk;
      EXPECT_EQ(t.age, static_cast<int>(g.num(tk + "_age"))) << tk;
      cv::Mat hist = g.mat(tk + "_hist");
      ASSERT_EQ(static_cast<int>(t.history.size()), hist.rows) << tk;
      for (int h = 0; h < hist.rows; ++h) {
        EXPECT_EQ(t.history[h].first, static_cast<int>(hist.at<double>(h, 0))) << tk;
        EXPECT_NEAR(t.history[h].second.x1, hist.at<double>(h, 1), 1e-9) << tk;
        EXPECT_NEAR(t.history[h].second.y2, hist.at<double>(h, 4), 1e-9) << tk;
      }
      const auto m = fcw::looming(t);
      ASSERT_EQ(m.has_value(), g.has(tk + "_loom_eta")) << tk;
      if (m) {
        EXPECT_NEAR(m->eta, g.num(tk + "_loom_eta"), 1e-9) << tk;
        EXPECT_NEAR(m->var, g.num(tk + "_loom_var"), 1e-9) << tk;
      }
    }
  }
}

}  // namespace

TEST(TrackerParity, ApproachingObject) { replay("approach"); }
TEST(TrackerParity, TwoObjectsKeepTheirOwnState) { replay("two"); }
TEST(TrackerParity, LostFrameIsNotInHistory) { replay("lost"); }
