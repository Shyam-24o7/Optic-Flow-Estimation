# Horn TTC Engine (HLS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the spec's PL engine (section 4.8) in Vitis HLS: one streaming pass over a frame pair that produces, for up to 16 boxes, the 10 Horn sums, the pixel count and the flow moments, bit-exact against the C++ host twin.

**Architecture:** `hls/ttc_engine/` holds the kernel (`ttc_engine.cpp`), a header shared with its testbench, and the testbench. The kernel reads the previous and current gray frames and the flow as one AXI-Stream of packed pixels, the box table over AXI-Lite, and writes the sums to memory. Internally: a 3-row line buffer of S = prev + curr, a 2-row buffer of Et and flow (to align with the Sobel window centre), a shared product unit, a 16-box mask, and per-box int64 accumulators. C-simulation is plain C++ compiled with the Vitis HLS headers, so it runs in WSL with g++; synthesis needs the Vitis device library.

**Tech Stack:** Vitis HLS 2024.2 (`ap_int.h`, `hls_stream.h`, `vitis-run --mode hls`), g++ 13 for C-simulation, the plan-1 Python reference and plan-2 C++ host for golden values.

**Spec:** `docs/superpowers/specs/2026-10-06-kv260-collision-warning-design.md` section 4.8 (and 4.6 for the integer definition).

## What runs where

| Task | Needs | Status on this PC |
|---|---|---|
| 1–3 golden data, kernel, C-simulation | g++, HLS headers | runs now (WSL) |
| 4 synthesis (II = 1, resources) | Vitis HLS with Zynq UltraScale+ device data | **blocked**: install "Zynq UltraScale+ MPSoC" devices for Vitis 2024.2 (AMD installer → Add Design Tools or Devices) |
| 5 host integration (XRT) and on-board speedup | KV260 | **blocked**: same as plan 2 Task 10 |

## Global Constraints

- Bit-exact: every Horn sum and count equals `fcw::hornSums` (C++) / `horn_sums` (Python) for the same frames, box and principal point. Flow moments are exact against a fixed-point twin (Task 3).
- Integer definition from spec 4.6: S = prev + curr (9 b); Ex, Ey = 3×3 Sobel of S on interior pixels (12 b); Et = curr − prev (9 b); G = x·Ex + y·Ey with x, y from the rounded principal point (21 b); 10 products, int64 accumulators.
- One pixel per clock in the main loop (`#pragma HLS PIPELINE II=1`); at most 16 boxes; frames up to 512×384.
- The host converts each box to its integer inner rectangle (`fcw::innerBox`) before the call, so the kernel only tests `x1 <= x < x2 && y1 <= y < y2`.
- Commits carry no Claude attribution (mandatory user instruction).

## Review Focus

1. **Window alignment:** gradients are for the pixel one row and one column behind the stream position; Et, flow and coordinates must refer to the same centre pixel. Golden case with a box touching the frame's interior edge.
2. **Overflow:** checkerboard frames over a 510×382 box reach 53-bit sums; int64 accumulators must not wrap.
3. **Box overlap:** a pixel inside several boxes adds to each of them.
4. **Threshold edge:** |Ex| + |Ey| exactly equal to the threshold is excluded (`>`), as in the reference.
5. **Empty boxes:** a box with no qualifying pixels returns all-zero sums and n = 0.

---

### Task 1: HLS golden vectors

**Files:** extend `tools/export_golden.py` (exporter `hls`), output `hls/ttc_engine/golden/` (binary frames, text expectations; small, committed).

Cases: random 64×48 frames with three overlapping boxes and threshold 0; the plan-1 approach scene at 512×384 with the scene box; the checkerboard at 512×384 with the near-full box; a flat frame (empty result). Each case writes `prev.bin`, `curr.bin` (uint8, row-major), `boxes.txt` (one `x1 y1 x2 y2` integer inner rectangle per line), `params.txt` (`width height px py threshold`), `expected.txt` (per box: 10 sums and n, decimal int64), all computed by the Python reference.

### Task 2: Kernel (Horn sums) with C-simulation

**Files:** `hls/ttc_engine/src/ttc_engine.hpp`, `hls/ttc_engine/src/ttc_engine.cpp`, `hls/ttc_engine/tb/tb_ttc_engine.cpp`, `hls/ttc_engine/Makefile` (csim target: g++ with `-I$(XILINX_HLS_INCLUDE)`), `hls/ttc_engine/hls_config.cfg` (for Task 4).

**Interface:**
```cpp
struct Pixel { ap_uint<8> prev, curr; ap_int<16> u, v; };       // flow in 1/64 px (Task 3)
struct BoxRect { ap_uint<10> x1, y1, x2, y2; };
void ttc_engine(hls::stream<Pixel>& in, const BoxRect boxes[16], ap_uint<5> n_boxes,
                ap_uint<10> width, ap_uint<10> height, ap_int<11> px, ap_int<11> py, ap_uint<13> threshold,
                ap_int<64> sums[16][kTerms]);
```
`kTerms` is 11 in this task (10 Horn sums + count) and grows to 23 in Task 3.

Testbench: for every golden case, stream the frames, call the kernel, compare every sum and count exactly; print the first mismatch with box, term and both values; return non-zero on any mismatch.

### Task 3: Fixed-point flow moments

**Files:** `collision_avoidance/fcw/divergence.py` (`flow_moments_fixed`), `tests/fcw/test_divergence.py`, the C++ twin in `host/src/divergence.cpp`, the kernel and testbench.

Flow enters the engine as int16 in 1/64 px (±512 px range). `flow_moments_fixed(flow_q, box, pp)` sums the 12 moments in integers (x, y relative to the rounded principal point); `affine_from_moments` accepts them after scaling u, v terms by 1/64. Tests: the fixed-point divergence is within 1% of the float one on the plan-1 divergence cases; the C++ twin equals Python exactly; the kernel equals both exactly.

### Task 4 (blocked: device data): Synthesis

`vitis-run --mode hls --csynth --config hls_config.cfg` for part `xck26-sfvc784-2LV-c` at 300 MHz. Pass: main loop II = 1; latency ≈ W·H + small constant cycles per call; DSP ≈ 19–32, LUT within the spec's estimate. Results go in `docs/superpowers/reports/<date>-horn-engine-synth.md`.

### Task 5 (blocked: board): Host integration

XRT runner `host/src/backends/board/xrt_ttc_engine.cpp` replacing `hornSums` (and, behind a flag, `flowMoments`) for boxes grouped by pyramid level; on-board check that warnings equal the CPU path and measurement of the perception-time change.
