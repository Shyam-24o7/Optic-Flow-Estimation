// Parity: fcw Horn direct TTC vs collision_avoidance/fcw/horn.py (golden file horn.yml.gz).
// The integer sums are the golden model for the HLS engine (plan 3): they must match exactly.
#include <gtest/gtest.h>

#include <cmath>

#include "fcw/horn.hpp"
#include "golden.hpp"

namespace {
fcw::Box box(const cv::Mat& m) { return {m.at<double>(0, 0), m.at<double>(0, 1), m.at<double>(0, 2), m.at<double>(0, 3)}; }
}  // namespace

TEST(HornParity, EveryCase) {
  golden::File g("horn");
  const double dt = g.num("dt");
  for (int i = 0; i < static_cast<int>(g.num("cases")); ++i) {
    const std::string k = "c" + std::to_string(i);
    SCOPED_TRACE(g.str(k + "_name"));
    fcw::HornConfig cfg;
    cfg.shrink = g.num(k + "_shrink");
    cfg.grad_threshold_l1 = static_cast<int>(g.num(k + "_thr"));
    const cv::Mat prev = g.image(k + "_prev"), curr = g.image(k + "_curr");
    const cv::Mat pp = g.mat(k + "_pp");
    const cv::Point2d principal(pp.at<double>(0, 0), pp.at<double>(0, 1));
    const int level = static_cast<int>(g.num(k + "_level"));
    std::optional<fcw::Measurement> m;
    if (level == 0) {
      const auto sums = fcw::hornSums(prev, curr, box(g.mat(k + "_box")), principal, cfg);
      ASSERT_EQ(sums.has_value(), g.num(k + "_has_sums") != 0);
      if (sums) {
        for (int j = 0; j < 10; ++j) EXPECT_EQ(sums->s[j], g.i64(k + "_sum" + std::to_string(j))) << "term " << fcw::kHornTerms[j];
        EXPECT_EQ(sums->n, g.i64(k + "_n"));
        m = fcw::hornSolve(*sums, dt, cfg);
      }
    } else {
      m = fcw::hornAtLevel(prev, curr, box(g.mat(k + "_box")), principal, level, dt, cfg);
    }
    ASSERT_EQ(m.has_value(), g.num(k + "_has_m") != 0);
    if (m) {
      EXPECT_NEAR(m->eta, g.num(k + "_eta"), 1e-9 * (1 + std::abs(g.num(k + "_eta"))));
      EXPECT_NEAR(m->var, g.num(k + "_var"), 1e-9 * (1 + g.num(k + "_var")));
    }
  }
}

TEST(HornParity, LevelChoice) {
  golden::File g("horn");
  const cv::Mat in = g.mat("levels_in"), out = g.mat("levels_out");
  for (int r = 0; r < in.rows; ++r) {
    const double e = in.at<double>(r, 0);
    const std::optional<double> eta = std::isnan(e) ? std::nullopt : std::optional<double>(e);
    const fcw::Box b{in.at<double>(r, 1), in.at<double>(r, 2), in.at<double>(r, 3), in.at<double>(r, 4)};
    EXPECT_EQ(fcw::chooseLevel(eta, b, in.at<double>(r, 5), g.num("dt")), static_cast<int>(out.at<double>(r, 0))) << "row " << r;
  }
}
