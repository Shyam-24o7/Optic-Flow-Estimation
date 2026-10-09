#include "ttc_engine.hpp"

// Raster scan. At stream position (x, y) the 3x3 window of S = prev + curr is complete for the
// centre pixel (x - 1, y - 1); Et and the coordinates are taken for that same centre.
void ttc_engine(hls::stream<Pixel>& in, const BoxRect boxes[kMaxBoxes], ap_uint<5> n_boxes, ap_uint<10> width,
                ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                ap_int<64> sums[kMaxBoxes][kTerms]) {
#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE s_axilite port = boxes bundle = control
#pragma HLS INTERFACE s_axilite port = n_boxes bundle = control
#pragma HLS INTERFACE s_axilite port = width bundle = control
#pragma HLS INTERFACE s_axilite port = height bundle = control
#pragma HLS INTERFACE s_axilite port = px bundle = control
#pragma HLS INTERFACE s_axilite port = py bundle = control
#pragma HLS INTERFACE s_axilite port = threshold bundle = control
#pragma HLS INTERFACE m_axi port = sums offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = return bundle = control

  BoxRect box[kMaxBoxes];
#pragma HLS ARRAY_PARTITION variable = box complete
  for (int b = 0; b < kMaxBoxes; ++b) box[b] = boxes[b];

  ap_int<64> acc[kMaxBoxes][kTerms];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  for (int b = 0; b < kMaxBoxes; ++b)
    for (int k = 0; k < kTerms; ++k) acc[b][k] = 0;

  ap_uint<9> line_s[2][kMaxWidth];  // S for rows y-2 and y-1
#pragma HLS ARRAY_PARTITION variable = line_s complete dim = 1
  ap_int<9> line_d[kMaxWidth];      // Et = curr - prev for row y-1
  ap_uint<9> win[3][3];             // window of S: [row y-2, y-1, y][column x-2, x-1, x]
#pragma HLS ARRAY_PARTITION variable = win complete dim = 0
  ap_int<9> d_centre = 0;           // Et at (x-1, y-1)

rows:
  for (int y = 0; y < height; ++y) {
  cols:
    for (int x = 0; x < width; ++x) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512
      const Pixel p = in.read();
      const ap_uint<9> s_new = p.prev + p.curr;
      const ap_int<9> d_new = ap_int<9>(p.curr) - ap_int<9>(p.prev);

      const ap_uint<9> s_top = line_s[0][x], s_mid = line_s[1][x];
      line_s[0][x] = s_mid;
      line_s[1][x] = s_new;
      const ap_int<9> d_above = line_d[x];  // row y-1, column x
      line_d[x] = d_new;

      for (int r = 0; r < 3; ++r) {
        win[r][0] = win[r][1];
        win[r][1] = win[r][2];
      }
      win[0][2] = s_top;
      win[1][2] = s_mid;
      win[2][2] = s_new;

      const ap_int<9> et = d_centre;  // row y-1, column x-1 (saved last iteration)
      d_centre = d_above;
      if (y < 2 || x < 2) continue;

      const ap_int<12> ex = (ap_int<12>(win[0][2]) + 2 * win[1][2] + win[2][2]) - (ap_int<12>(win[0][0]) + 2 * win[1][0] + win[2][0]);
      const ap_int<12> ey = (ap_int<12>(win[2][0]) + 2 * win[2][1] + win[2][2]) - (ap_int<12>(win[0][0]) + 2 * win[0][1] + win[0][2]);
      const ap_uint<13> mag = (ex < 0 ? ap_int<13>(-ex) : ap_int<13>(ex)) + (ey < 0 ? ap_int<13>(-ey) : ap_int<13>(ey));
      if (mag <= threshold) continue;

      const int cx = x - 1, cy = y - 1;
      const ap_int<11> rx = cx - px, ry = cy - py;
      const ap_int<22> g = rx * ex + ry * ey;
      // Products computed once per pixel, shared by every box.
      ap_int<64> prod[kTerms];
#pragma HLS ARRAY_PARTITION variable = prod complete
      prod[0] = ap_int<64>(ex) * ex;
      prod[1] = ap_int<64>(ex) * ey;
      prod[2] = ap_int<64>(ex) * g;
      prod[3] = ap_int<64>(ey) * ey;
      prod[4] = ap_int<64>(ey) * g;
      prod[5] = ap_int<64>(g) * g;
      prod[6] = ap_int<64>(ex) * et;
      prod[7] = ap_int<64>(ey) * et;
      prod[8] = ap_int<64>(g) * et;
      prod[9] = ap_int<64>(et) * et;
      prod[10] = 1;
      for (int b = 0; b < kMaxBoxes; ++b) {
        const bool inside = b < n_boxes && cx >= box[b].x1 && cx < box[b].x2 && cy >= box[b].y1 && cy < box[b].y2;
        if (inside)
          for (int k = 0; k < kTerms; ++k) acc[b][k] += prod[k];
      }
    }
  }

  for (int b = 0; b < kMaxBoxes; ++b)
    for (int k = 0; k < kTerms; ++k) sums[b][k] = acc[b][k];
}
