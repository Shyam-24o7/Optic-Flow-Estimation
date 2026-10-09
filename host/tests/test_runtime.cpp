// T1-T4 runtime: ordering, back-pressure and the CLI summary line.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "fcw/runtime.hpp"

namespace {

// Synthetic camera: a random texture that drifts one pixel per frame.
fcw::CaptureSource syntheticSource(int frames) {
  auto i = std::make_shared<int>(0);
  cv::Mat tex(384 + frames, 512 + frames, CV_8UC3);
  cv::randu(tex, 0, 255);
  return [=]() -> std::optional<fcw::CapturedFrame> {
    if (*i >= frames) return std::nullopt;
    const int k = (*i)++;
    return fcw::CapturedFrame{tex(cv::Rect(k, 0, 512, 384)).clone(), k / 30.0};
  };
}

fcw::DetectorFn oneCar(int delay_ms = 0) {
  return [delay_ms](const cv::Mat&) {
    if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    return std::vector<fcw::Detection>{{{200, 150, 300, 230}, "car", 0.9}};
  };
}

}  // namespace

TEST(BoundedQueue, DropsOldestWhenFull) {
  fcw::BoundedQueue<int> q(2);
  EXPECT_FALSE(q.pushDropOldest(1));
  EXPECT_FALSE(q.pushDropOldest(2));
  EXPECT_TRUE(q.pushDropOldest(3));  // 1 dropped
  EXPECT_EQ(*q.pop(), 2);
  EXPECT_EQ(*q.pop(), 3);
  q.close();
  EXPECT_FALSE(q.pop().has_value());
}

TEST(Runtime, ProcessesEveryFrameInOrderWhenNotLive) {
  std::vector<int> indices;
  std::vector<double> times;
  fcw::RuntimeOptions opt;
  opt.live = false;
  fcw::Runtime rt(fcw::FcwConfig{}, opt, syntheticSource(25), oneCar(), fcw::disFlow(),
                  [&](const fcw::FcwFrame& f) { indices.push_back(f.index); times.push_back(f.time_s); });
  const fcw::RuntimeStats s = rt.run();
  EXPECT_EQ(s.captured, 25);
  EXPECT_EQ(s.processed, 25);
  EXPECT_EQ(s.dropped, 0);
  ASSERT_EQ(indices.size(), 25u);
  for (int i = 0; i < 25; ++i) EXPECT_EQ(indices[i], i);
  for (size_t i = 1; i < times.size(); ++i) EXPECT_GT(times[i], times[i - 1]);
}

TEST(Runtime, DropsAtCaptureUnderLoadAndStaysMonotonic) {
  std::vector<double> times;
  fcw::RuntimeOptions opt;
  opt.live = true;
  opt.capture_queue = 2;
  fcw::Runtime rt(fcw::FcwConfig{}, opt, syntheticSource(60), oneCar(15), fcw::disFlow(),
                  [&](const fcw::FcwFrame& f) { times.push_back(f.time_s); });
  const fcw::RuntimeStats s = rt.run();
  EXPECT_EQ(s.captured, 60);
  EXPECT_GT(s.dropped, 0);                         // the slow detector cannot keep up
  EXPECT_EQ(s.processed + s.dropped, s.captured);  // nothing lost silently
  EXPECT_LE(s.max_capture_queue, 2);               // bounded
  EXPECT_EQ(static_cast<int>(times.size()), s.processed);
  for (size_t i = 1; i < times.size(); ++i) EXPECT_GT(times[i], times[i - 1]);
}

TEST(Runtime, SummaryLineMatchesThePythonCli) {
  const std::vector<fcw::Level> levels{fcw::Level::None, fcw::Level::Warning, fcw::Level::Warning, fcw::Level::None, fcw::Level::Critical};
  EXPECT_EQ(fcw::summaryLine(levels, {50.0, 10.0, 20.0, 30.0, 40.0}),
            "frames 5 | mean 25.0 ms/frame | warnings raised 2 | critical frames 1");
}

TEST(Pipeline, ReportsTimePerPerceptionStep) {
  // Per-method timings feed the on-board latency histograms (spec section 8).
  auto source = syntheticSource(3);
  fcw::FcwPipeline pipe(fcw::FcwConfig{}, oneCar(), fcw::disFlow());
  fcw::FcwFrame last;
  while (auto f = source()) last = pipe.process(f->bgr, f->t);
  for (const char* key : {"flow", "detection", "track", "ego", "looming", "scale", "horn", "divergence", "fusion_course", "warn", "total"})
    EXPECT_TRUE(last.timings_ms.count(key)) << key;
}
