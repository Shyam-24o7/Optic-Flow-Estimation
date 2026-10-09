// Parity: fcw fusion / course check / warning FSM vs collision_avoidance/fcw/fusion.py and collision.py.
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>

#include "fcw/collision.hpp"
#include "fcw/fusion.hpp"
#include "golden.hpp"

namespace {

const char* kMethods[4] = {"looming", "scale", "horn", "divergence"};

int methodIndex(const std::string& m) {
  for (int i = 0; i < 4; ++i)
    if (m == kMethods[i]) return i;
  return -1;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string x; std::getline(ss, x, ',');) out.push_back(x);
  return out;
}

void expectNearOrNan(double actual, double expected, double tol, const std::string& what) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual)) << what;
  } else {
    EXPECT_NEAR(actual, expected, tol * (1 + std::abs(expected))) << what;
  }
}

}  // namespace

TEST(FusionParity, EveryScenario) {
  golden::File g("fusion");
  const double dt = g.num("dt");
  for (const std::string& name : split(g.str("scenarios"))) {
    SCOPED_TRACE(name);
    fcw::FusionConfig cfg;
    cfg.max_consecutive_rejects = static_cast<int>(g.num(name + "_max_rejects"));
    fcw::TtcFusion fusion(cfg);
    for (int i = 0; i < static_cast<int>(g.num(name + "_steps")); ++i) {
      const std::string k = name + "_s" + std::to_string(i);
      const cv::Mat in = g.mat(k + "_in");
      std::vector<fcw::Measurement> ms;
      for (int r = 0; r < in.rows; ++r) ms.push_back({kMethods[static_cast<int>(in.at<double>(r, 0))], in.at<double>(r, 1), in.at<double>(r, 2)});
      const auto est = fusion.update(1, i * dt, dt, ms);
      ASSERT_EQ(est.has_value(), g.num(k + "_has") != 0) << k;
      if (!est) continue;
      const cv::Mat out = g.mat(k + "_out");
      expectNearOrNan(est->eta, out.at<double>(0, 0), 1e-12, k + " eta");
      expectNearOrNan(est->eta_var, out.at<double>(0, 1), 1e-12, k + " var");
      expectNearOrNan(est->ttc_s ? *est->ttc_s : NAN, out.at<double>(0, 2), 1e-12, k + " ttc");
      expectNearOrNan(est->sigma_ttc_s ? *est->sigma_ttc_s : NAN, out.at<double>(0, 3), 1e-12, k + " sigma");
      const cv::Mat acc = g.mat(k + "_accepted"), rec = g.mat(k + "_recent");
      ASSERT_EQ(static_cast<int>(est->accepted.size()), acc.rows) << k;
      for (int r = 0; r < acc.rows; ++r) EXPECT_EQ(methodIndex(est->accepted[r].method), static_cast<int>(acc.at<double>(r, 0))) << k;
      std::vector<int> recent;
      for (const auto& m : est->recent_methods) recent.push_back(methodIndex(m));
      std::sort(recent.begin(), recent.end());
      ASSERT_EQ(static_cast<int>(recent.size()), rec.rows) << k;
      for (int r = 0; r < rec.rows; ++r) EXPECT_EQ(recent[r], static_cast<int>(rec.at<double>(r, 0))) << k;
    }
  }
}

TEST(CollisionParity, CourseCheckSequence) {
  golden::File g("collision");
  const auto classes = split(g.str("classes"));
  fcw::CourseChecker checker;
  for (int i = 0; i < static_cast<int>(g.num("course_steps")); ++i) {
    const std::string k = "course_s" + std::to_string(i);
    const cv::Mat in = g.mat(k + "_in"), out = g.mat(k + "_out");
    const double ttc = in.at<double>(0, 9);
    const auto res = checker.update(static_cast<int>(in.at<double>(0, 0)), in.at<double>(0, 1),
                                    {in.at<double>(0, 2), in.at<double>(0, 3), in.at<double>(0, 4), in.at<double>(0, 5)},
                                    classes.at(static_cast<int>(in.at<double>(0, 6))), in.at<double>(0, 7), in.at<double>(0, 8),
                                    std::isnan(ttc) ? std::nullopt : std::optional<double>(ttc), in.at<double>(0, 10), cv::Size(512, 384),
                                    cv::Matx33d(g.mat(k + "_rot")), 256.0);
    EXPECT_EQ(res.on_course, out.at<double>(0, 0) != 0) << k;
    EXPECT_NEAR(res.r, out.at<double>(0, 1), 1e-12) << k;
    expectNearOrNan(res.r_contact ? *res.r_contact : NAN, out.at<double>(0, 2), 1e-9, k + " r_contact");
    EXPECT_NEAR(res.threshold, out.at<double>(0, 3), 1e-12) << k;
    EXPECT_EQ(res.in_path, out.at<double>(0, 4) != 0) << k;
    expectNearOrNan(res.entry_speed_mps ? *res.entry_speed_mps : NAN, out.at<double>(0, 5), 1e-9 * (1 + std::abs(out.at<double>(0, 5))), k + " entry_speed");
  }
}

TEST(CollisionParity, WarningFsmSequence) {
  golden::File g("collision");
  const cv::Mat in = g.mat("fsm_in"), out = g.mat("fsm_out");
  fcw::WarningFsm fsm;
  for (int r = 0; r < in.rows; ++r) {
    const double ttc = in.at<double>(r, 1), sigma = in.at<double>(r, 2);
    const fcw::Level level = fsm.step(in.at<double>(r, 0) != 0, std::isnan(ttc) ? std::nullopt : std::optional<double>(ttc),
                                      std::isnan(sigma) ? std::nullopt : std::optional<double>(sigma), static_cast<int>(in.at<double>(r, 3)));
    EXPECT_EQ(static_cast<int>(level), static_cast<int>(out.at<double>(r, 0))) << "step " << r;
  }
}
