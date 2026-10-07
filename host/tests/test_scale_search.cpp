// Parity: fcw scale search vs collision_avoidance/fcw/scale_search.py (golden file scale.yml.gz).
#include <gtest/gtest.h>

#include <cmath>

#include "fcw/scale_search.hpp"
#include "golden.hpp"

namespace {
fcw::Box box(const cv::Mat& m) { return {m.at<double>(0, 0), m.at<double>(0, 1), m.at<double>(0, 2), m.at<double>(0, 3)}; }
}  // namespace

TEST(ScaleParity, EveryCase) {
  golden::File g("scale");
  const double dt = g.num("dt");
  for (int i = 0; i < static_cast<int>(g.num("cases")); ++i) {
    const std::string k = "c" + std::to_string(i);
    SCOPED_TRACE(g.str(k + "_name"));
    const cv::Mat gt = g.image(k + "_gray_t"), gtk = g.image(k + "_gray_tk");
    const fcw::Box bt = box(g.mat(k + "_box_t")), btk = box(g.mat(k + "_box_tk"));
    const int gap = static_cast<int>(g.num(k + "_k"));
    const auto found = fcw::searchScale(gt, gtk, bt, btk);
    ASSERT_EQ(found.has_value(), g.num(k + "_found") != 0);
    if (found) {
      // Flat NCC peaks (static object) amplify float32 (numpy) vs double score differences in the
      // parabola fit; 1e-5 is 0.25% of one fine step, i.e. < 1e-4 1/s in eta.
      EXPECT_NEAR(found->first, g.num(k + "_s"), 1e-5);
      EXPECT_NEAR(found->second, g.num(k + "_peak"), 1e-6);
    }
    const auto m = fcw::scaleTtc(gt, gtk, bt, btk, gap, dt);
    ASSERT_EQ(m.has_value(), g.num(k + "_has_m") != 0);
    if (m) {
      EXPECT_EQ(m->method, "scale");
      EXPECT_NEAR(m->eta, g.num(k + "_eta"), 1e-4);
      EXPECT_NEAR(m->var, g.num(k + "_var"), 1e-6 * (1 + g.num(k + "_var")));
    }
  }
}

TEST(ScaleParity, GapChoice) {
  golden::File g("scale");
  const cv::Mat in = g.mat("gaps_in"), out = g.mat("gaps_out");
  for (int r = 0; r < in.rows; ++r) {
    const double e = in.at<double>(r, 0);
    const std::optional<double> eta = std::isnan(e) ? std::nullopt : std::optional<double>(e);
    EXPECT_EQ(fcw::chooseGap(eta, g.num("dt"), static_cast<int>(in.at<double>(r, 1))), static_cast<int>(out.at<double>(r, 0))) << "row " << r;
  }
}
