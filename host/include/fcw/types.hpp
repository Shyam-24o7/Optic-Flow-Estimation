// Shared value types of the FCW host app. Mirrors collision_avoidance/fcw (Python reference).
#pragma once

#include <string>

namespace fcw {

// (x1, y1, x2, y2) in pixels of the 512x384 processing frame.
struct Box {
  double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  double width() const { return x2 - x1; }
  double height() const { return y2 - y1; }
  double cx() const { return (x1 + x2) / 2; }
  double cy() const { return (y1 + y2) / 2; }
};

struct Detection {
  Box box;
  std::string cls;
  double conf = 0;
};

// Output of every TTC method: inverse TTC (1/s) and its variance (python: measurement.py).
struct Measurement {
  std::string method;  // "looming" | "scale" | "horn" | "divergence"
  double eta = 0;
  double var = 0;
};

}  // namespace fcw
