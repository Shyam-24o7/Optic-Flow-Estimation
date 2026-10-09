// C-simulation testbench: every golden case from tools/export_golden.py --only hls must match exactly.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/ttc_engine.hpp"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "golden"
#endif

static std::vector<uint8_t> readBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static int runCase(const std::string& name) {
  const std::string dir = std::string(GOLDEN_DIR) + "/" + name + "/";
  int w, h, px, py, thr;
  std::ifstream(dir + "params.txt") >> w >> h >> px >> py >> thr;
  const std::vector<uint8_t> prev = readBytes(dir + "prev.bin"), curr = readBytes(dir + "curr.bin");
  if (static_cast<int>(prev.size()) != w * h || static_cast<int>(curr.size()) != w * h) {
    std::printf("%s: frame size mismatch\n", name.c_str());
    return 1;
  }
  BoxRect boxes[kMaxBoxes] = {};
  int n = 0;
  std::ifstream bf(dir + "boxes.txt");
  for (int x1, y1, x2, y2; bf >> x1 >> y1 >> x2 >> y2;) boxes[n++] = {x1, y1, x2, y2};

  hls::stream<Pixel> in;
  for (int i = 0; i < w * h; ++i) in.write({prev[i], curr[i], 0, 0});
  static ap_int<64> sums[kMaxBoxes][kTerms];
  ttc_engine(in, boxes, n, w, h, px, py, thr, sums);

  std::ifstream ef(dir + "expected.txt");
  int failures = 0;
  for (int b = 0; b < n; ++b) {
    for (int k = 0; k < kTerms; ++k) {
      long long expected;
      ef >> expected;
      const long long got = sums[b][k].to_int64();
      if (got != expected && failures++ < 5)
        std::printf("%s box %d term %d: got %lld expected %lld\n", name.c_str(), b, k, got, expected);
    }
  }
  std::printf("%-18s %d boxes  %s\n", name.c_str(), n, failures ? "FAIL" : "exact");
  return failures ? 1 : 0;
}

int main() {
  int failed = 0;
  for (const char* c : {"random_overlap", "random_threshold", "approach", "checker", "flat"}) failed += runCase(c);
  std::printf(failed ? "C-SIM FAILED (%d cases)\n" : "C-SIM PASSED\n", failed);
  return failed ? 1 : 0;
}
