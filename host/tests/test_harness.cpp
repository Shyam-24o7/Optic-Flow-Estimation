#include <gtest/gtest.h>

#include "golden.hpp"

TEST(Harness, ReadsMatricesNumbersAndInt64) {
  golden::File g("harness");
  cv::Mat m = g.mat("m");
  ASSERT_EQ(m.rows, 2);
  ASSERT_EQ(m.cols, 3);
  EXPECT_DOUBLE_EQ(m.at<double>(1, 2), 6.5);
  EXPECT_DOUBLE_EQ(g.num("x"), 3.25);
  EXPECT_EQ(g.i64("big"), 6000000000LL);  // above 2^32: must not pass through a 32-bit int
}
