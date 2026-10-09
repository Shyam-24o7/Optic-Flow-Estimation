// Horn TTC engine (spec section 4.8): one streaming pass over a frame pair,
// per-box Horn sums for up to 16 boxes. Bit-exact with fcw::hornSums (host/src/horn.cpp)
// and collision_avoidance/fcw/horn.py.
#pragma once

#include <ap_int.h>
#include <hls_stream.h>

constexpr int kMaxWidth = 512;
constexpr int kMaxBoxes = 16;
// 10 Horn sums (Ex^2, ExEy, ExG, Ey^2, EyG, G^2, ExEt, EyEt, GEt, Et^2) + pixel count.
constexpr int kTerms = 11;

struct Pixel {
  ap_uint<8> prev;
  ap_uint<8> curr;
  ap_int<16> u;  // flow in 1/64 px (plan 3, Task 3)
  ap_int<16> v;
};

// Integer inner rectangle [x1, x2) x [y1, y2), from fcw::innerBox on the host.
struct BoxRect {
  ap_uint<10> x1, y1, x2, y2;
};

void ttc_engine(hls::stream<Pixel>& in, const BoxRect boxes[kMaxBoxes], ap_uint<5> n_boxes, ap_uint<10> width,
                ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                ap_int<64> sums[kMaxBoxes][kTerms]);
