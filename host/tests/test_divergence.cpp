// Parity: fcw divergence vs collision_avoidance/fcw/divergence.py (golden file divergence.yml.gz).
#include <gtest/gtest.h>

#include "fcw/divergence.hpp"
#include "golden.hpp"

TEST(DivergenceParity, EveryCase) {
  golden::File g("divergence");
  const double dt = g.num("dt");
  const cv::Mat ppm = g.mat("pp"), bm = g.mat("box"), win = g.mat("window");
  const cv::Point2d pp(ppm.at<double>(0, 0), ppm.at<double>(0, 1));
  const fcw::Box box{bm.at<double>(0, 0), bm.at<double>(0, 1), bm.at<double>(0, 2), bm.at<double>(0, 3)};
  const cv::Rect window(cv::Point(static_cast<int>(win.at<double>(0, 0)), static_cast<int>(win.at<double>(0, 1))),
                        cv::Point(static_cast<int>(win.at<double>(0, 2)), static_cast<int>(win.at<double>(0, 3))));
  for (int i = 0; i < static_cast<int>(g.num("cases")); ++i) {
    const std::string k = "c" + std::to_string(i);
    SCOPED_TRACE(g.str(k + "_name"));
    cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
    g.mat(k + "_flow").copyTo(flow(window));
    const double rot = g.num(k + "_rotdiv");

    const auto m = fcw::flowMoments(flow, box, pp);
    ASSERT_TRUE(m.has_value());
    const cv::Mat gm = g.mat(k + "_moments");
    for (int j = 0; j < 12; ++j) EXPECT_NEAR((*m)[j], gm.at<double>(0, j), 1e-9 * (1 + std::abs(gm.at<double>(0, j)))) << "term " << j;
    const auto plain = fcw::divergenceTtc(m, box, rot, 1, dt);
    ASSERT_TRUE(plain.has_value());
    EXPECT_NEAR(plain->eta, g.num(k + "_plain_eta"), 1e-9);
    EXPECT_NEAR(plain->var, g.num(k + "_plain_var"), 1e-12);

    const cv::Mat s = g.mat(k + "_samples");
    std::vector<std::array<int, 3>> samples;
    for (int r = 0; r < s.rows; ++r) samples.push_back({static_cast<int>(s.at<double>(r, 0)), static_cast<int>(s.at<double>(r, 1)), static_cast<int>(s.at<double>(r, 2))});
    const cv::Mat mask = fcw::robustMask(flow, box, pp, {}, samples);
    cv::Mat gmask = g.mat(k + "_mask"), mine;
    mask(window).convertTo(mine, CV_8U);
    EXPECT_EQ(cv::countNonZero(mine != gmask), 0);
    const auto robust = fcw::divergenceTtc(fcw::flowMoments(flow, box, pp, mask), box, rot, 1, dt);
    ASSERT_TRUE(robust.has_value());
    EXPECT_NEAR(robust->eta, g.num(k + "_robust_eta"), 1e-9);
  }
}

TEST(DivergenceParity, ProductionRansacRejectsTheBackgroundLeak) {
  // Own sampling (std::mt19937), not Python's: must still find the object's flow.
  golden::File g("divergence");
  const cv::Mat ppm = g.mat("pp"), bm = g.mat("box"), win = g.mat("window");
  const cv::Point2d pp(ppm.at<double>(0, 0), ppm.at<double>(0, 1));
  const fcw::Box box{bm.at<double>(0, 0), bm.at<double>(0, 1), bm.at<double>(0, 2), bm.at<double>(0, 3)};
  const cv::Rect window(cv::Point(static_cast<int>(win.at<double>(0, 0)), static_cast<int>(win.at<double>(0, 1))),
                        cv::Point(static_cast<int>(win.at<double>(0, 2)), static_cast<int>(win.at<double>(0, 3))));
  cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
  g.mat("c2_flow").copyTo(flow(window));  // case 2: background leak
  const auto m = fcw::divergenceTtc(fcw::flowMoments(flow, box, pp, fcw::robustMask(flow, box, pp)), box, 0.0, 1, g.num("dt"));
  ASSERT_TRUE(m.has_value());
  EXPECT_NEAR(m->eta, g.num("c2_robust_eta"), 1e-6);
}

TEST(DivergenceParity, EmptyBoxHasNoMoments) {
  cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
  EXPECT_FALSE(fcw::flowMoments(flow, {600, 10, 700, 50}, {256, 192}).has_value());
  EXPECT_FALSE(fcw::divergenceTtc(std::nullopt, {0, 0, 10, 10}, 0.0, 1, 1 / 30.0).has_value());
}
