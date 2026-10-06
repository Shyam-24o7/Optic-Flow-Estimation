# KV260 Forward Collision Warning: Design

- **Date:** 2026-10-06
- **Status:** approved 2026-10-06; revised the same day with findings from prototyping the algorithm core (see section 12)
- **Diagrams:** https://claude.ai/artifact/LmHw3Nxg4rDhFAWXeXXZNY (high- and low-level architecture, dataflow, timing, engine, warning FSM)
- **Builds on:** `shadowPunch/Optic-Flow-Estimation` after PR #1 (reconstructed pipeline, Vitis AI results). This clone predates that PR.

## 1. Goal and scope

Warn the driver of an impending forward collision using a single camera and an AMD Kria KV260. The deliverable is a contest demo: a live, convincing run on the board with measured FPS, latency and TTC accuracy.

**In scope:** detection, tracking, camera-rotation estimation, per-object time to collision (TTC), a collision-course check, two warning levels, an on-screen overlay, and one custom HLS kernel.

**Out of scope:** braking or steering control, stereo, LiDAR, IMU or CAN input, curved-path prediction, night-specific tuning.

## 2. Constraints and assumptions

| Item | Value | Source |
|---|---|---|
| Board | Kria KV260 (ZU5EV, 4× Cortex-A53, 4 GB DDR4) | stated |
| Platform | road vehicle, forward camera | stated |
| Sensors | camera only; no IMU, no CAN | stated |
| Goal | contest demo | stated |
| Detector | YOLO; YOLOv9t-Hardswish reused from upstream | stated (YOLO), assumed (v9t, unobjected) |
| Flow | Vitis Vision LK first; learned flow (PWC-DC-Net QAT) as a stretch goal | stated |
| Processing resolution | 512×384 | assumed, matches upstream |
| Frame rate | 30 FPS input | assumed |
| Latency target | under 100 ms from capture to warning | assumed |
| Camera intrinsics K | known (checkerboard calibration, or dataset-provided) | required |

## 3. Architecture

| Where | Component | Rate |
|---|---|---|
| PL | Preprocessing (Vitis Vision): resize to 512×384, RGB tensor, gray image + pyramid | every frame |
| PL (DPU B4096) | YOLOv9t-Hardswish, INT8, one DPU subgraph | every frame |
| PL | Vitis Vision dense pyramidal LK, 512×384 | every frame |
| PL (P2) | Horn TTC engine (custom HLS) | every frame |
| PS T1 | capture (USB camera or file), decode into DDR, timestamp | every frame |
| PS T2 | accelerator runner: VART for the DPU, XRT for PL kernels | every frame |
| PS T3 | decode + NMS, tracker, ego-rotation, TTC methods, fusion, course check, warning FSM | every frame |
| PS T4 | overlay and output (HDMI/DP) | every frame |

All exchanges between PS and PL go through shared zero-copy XRT buffers in DDR. Threads pass frames along queues, so frames N, N+1 and N+2 are in flight at once.

**Placement rule.** Work that touches every pixel every frame goes to the PL or DPU. Work that scales with the number of tracked objects stays on the ARM cores. Anything else moves to the PL only if P1 measurements show it breaking the CPU budget. The first candidate would be the scale-ratio search.

## 4. Components

### 4.1 Preprocessing (PL)
Reads the decoded frame from DDR. Writes:
- the RGB tensor for the DPU (512×384×3 INT8, 576 KiB);
- the gray frame into a 9-slot ring buffer (current frame + 8 past frames, 9 × 192 KiB ≈ 1.7 MiB);
- the gray pyramid levels used by LK and the Horn engine.

### 4.2 Detector (DPU)
- **Model:** YOLOv9t with SiLU replaced by Hardswish, compiled for `DPUCZDX8G_ISA1_B4096` as one DPU subgraph (upstream result: 0.317 COCO mAP50-95).
- **On the ARM cores:** decode and NMS.
- **Collision-relevant classes:** car, truck, bus, motorcycle, bicycle, person.

### 4.3 Optical flow (PL)
- **Producer:** Vitis Vision dense pyramidal LK on consecutive gray frames at 512×384, filling `FlowField` with 16-bit fixed-point u, v.
- **If P0 shows LK can't sustain 30 FPS at 512×384:** run it at 256×192 and scale the vectors.
- **P3 stretch goal:** PWC-DC-Net with QAT fills the same `FlowField` interface (section 9).

### 4.4 Tracker (PS)
- **Kalman state:** [cx, cy, σ = ln√(w·h), aspect, ċx, ċy, σ̇].
- **Measurement:** [cx, cy, σ, aspect].
- **Association:** IoU cost plus Hungarian assignment, IoU ≥ 0.3. A track is deleted after 30 lost frames.
- **Prediction** uses the track's own state. This fixes the original `predict_new_location` bug, which looked up the wrong track.
- **TTC set:** at most 16 tracks receive TTC estimation each frame, chosen in priority order: on-course tracks first, then by box area, largest first.

### 4.5 Ego-rotation (PS)
1. **Sample** LK flow on a 32×24 grid (every 16 px, about 768 points). Skip points inside the previous frame's tracked boxes (enlarged by 10%) and inside the hood mask.
2. **Fit** with `cv::findEssentialMat` (RANSAC, 1 px threshold, 0.999 confidence) followed by `cv::recoverPose`, giving R and the translation direction t̂. FOE = K·t̂ when t̂ has a positive z component.
3. **Stationary case (model selection):** also fit a homography (RANSAC). If K⁻¹·H·K is within 1% of a rotation and the homography explains at least 90% as many points as the essential matrix, the camera only rotated: set `stationary`, take R from the homography, report no FOE. (The essential matrix is degenerate without translation.)
4. **Low confidence:** confidence is the essential matrix's RANSAC inlier share. (`recoverPose`'s own mask also drops points beyond 50 baselines, i.e. far background, so it is not used.) Below 0.5, hold the previous estimate and mark it low-confidence.
5. **Temporal filter:** reject a new R whose angle differs from the last valid one by more than 1° per frame; smooth the FOE with an EMA (weight 0.5). FOE = K·t̂ / t̂_z, valid when |t̂_z| ≥ 0.5.
6. **Output:** `EgoMotion{R, ω, FOE, inlier_ratio, stationary, valid}`.

Only rotation is removed. Forward translation is the collision signal and is kept.

### 4.6 TTC methods (each produces η = 1/τ in s⁻¹ with a variance)

Δt is taken from frame timestamps. k is the frame gap.

1. **Looming:** η_L = σ̇ from the tracker, with variance taken from its covariance.
2. **Scale-ratio search:**
   - **Frame gap:** k is chosen adaptively from {1, 2, 4, 8} so the expected expansion is at least 2%. Far objects use larger k.
   - **Crops:** the track's box, shrunk by 10%, from gray frames t and t−k, each resampled to 32×32.
   - **Search:** s over [0.95, 1.25]: 16 coarse steps of 0.02, then 9 fine steps of 0.004 around the best coarse value. Score each with zero-mean NCC and fit a parabola for sub-step precision.
   - **Result:** τ = kΔt / (s* − 1). Confidence comes from the NCC peak's curvature and height.
3. **Horn direct TTC:**
   - **Model:** A·Ex + B·Ey + C·G + Et = 0 with G = x·Ex + y·Ey. x, y are measured from the principal point; moving the origin only reparameterizes A and B.
   - **Integer definition (the HLS engine must match it bit for bit):** S = I_{t−1} + I_t; Ex, Ey = 3×3 Sobel of S on interior pixels (12 b); Et = I_t − I_{t−1} (9 b). Sobel of the frame sum is 16× the mean-intensity derivative, so the fitted C′ gives C = 16·C′.
   - **Solve:** 10 per-box sums (the 9 normal-equation terms plus ΣEt², which gives the residual and the variance) → C, and η_H = C/Δt.
   - **Pyramid level:** each box uses the level where its expected motion stays under about 1 px.
   - **Masking and gates:** pixels with |∇E| ≤ threshold are skipped. The estimate is dropped when the system is ill-conditioned or the residual is high.
4. **LK divergence:**
   - **Fit:** an affine flow model u = a₀ + a₁x + a₂y, v = b₀ + b₁x + b₂y over the box, using the 11 flow-moment sums. div = a₁ + b₂.
   - **De-rotation:** subtract the rotation's divergence at the box centre, computed exactly from H = K·R·K⁻¹. For small rotations it equals 3(x_n·ω_y − y_n·ω_x) in normalized coordinates, with ω the rotation vector of R (prev → curr). The earlier draft used the opposite, camera-rotation sign convention.
   - **Result:** η_D = div / (2kΔt), with k = 1 for LK.
   - **Outlier rejection:** RANSAC on the affine model (64 hypotheses on a stride-2 subsample, 1 px inlier threshold), then least squares on the inliers. An iterated least-squares refit was tested and fails: background pixels drag the first fit before any residual test can reject them. In P2 the engine is single-pass, so it masks pixels that deviate from the previous frame's fit, or the ARM cores run the RANSAC on a subsample if the DSP budget is short.

### 4.7 Fusion (PS)
- **Filter:** a per-track EKF on η.
- **Motion model:** constant closing speed, η′ = η / (1 − ηΔt). Process noise covers braking and acceleration.
- **Measurement updates:** each available η_i with its variance. The variance is the method's own estimate times a per-method factor calibrated in P1 on EvTTC, with a floor for model error: σ ≥ rel·|η| + 0.02 s⁻¹, rel = 0.05 (scale), 0.15 (Horn), 0.30 (divergence), 0 (looming). Without the floor, Horn's tiny least-squares variance let its +10–30% bias (seen on synthetic scenes) dominate the fused TTC.
- **Re-initialisation:** after 5 consecutive frames whose measurements are all gated out, the filter restarts from the current measurements.
- **Gate:** a Mahalanobis test rejects outlier measurements.
- **Output:** `TtcEstimate`, with τ = 1/η when η > 0.05 s⁻¹ (otherwise "not approaching"), plus σ_τ, the four η values and which methods contributed.

### 4.8 Horn TTC engine (PL, P2)
- **Inputs:**
  - gray t and gray t−1 (AXI-Stream);
  - flow u, v (AXI-Stream);
  - box table (AXI-Lite): up to 16 boxes with their pyramid levels.
- **Pipeline at II = 1:**
  1. Line buffers; S = I_{t−1} + I_t; 3×3 Sobel of S → Ex, Ey (12 b); Et = I_t − I_{t−1} (9 b).
  2. Pixel counter → x, y relative to the principal point (10 b).
  3. G = x·Ex + y·Ey (21 b).
  4. Shared product unit forms 22 terms per pixel, using about 19 multipliers:
     - Horn (10): Ex², ExEy, ExG, Ey², EyG, G², ExEt, EyEt, GEt, Et²;
     - flow moments (11): x, y, x², xy, y², u, v, xu, yu, xv, yv;
     - pixel count (1).
  5. Box mask unit: for each box, the pixel is inside it, |∇E| exceeds the threshold (Horn terms) and the flow is valid (moment terms).
  6. Accumulator bank: 16 boxes × 22 int64 adders. Products are shared, so each extra box costs adders only.
- **Output:** 16 × 22 × 8 B = 2,816 B per frame to DDR over m_axi. The ARM cores solve the 3×3 systems.
- **Precision:** |Ex|, |Ey| ≤ 4·510 = 2040; |G| ≤ 448·2040 = 913,920; the largest per-pixel product is G² ≤ 8.4×10¹¹ (40 b). A box has at most 196,608 pixels (18 b), so sums reach 58 b; int64 is safe.
- **Resources (estimate):** about 19–32 DSPs, about 21K LUTs for the accumulators. If P0 shows the fabric is tight: 8 boxes, or 48-bit partial sums.
- **Reuses** upstream's `hls/image_derivative` Sobel after fixing its row/column offset (it emits the window centred on (x−1, y−2)).
- **Before P2:** a bit-exact fixed-point C++ twin runs in T3 and is the golden reference.

### 4.9 Course check (PS)
- **Heading:** the FOE x-coordinate x_F (smoothed). When the car is stationary or the FOE is invalid, use the principal point.
- **Lateral offset in object widths:** r = (x_c − x_F) / w = X / W_obj, which is independent of distance. ṙ comes from a least-squares line over the last 0.5 s, fitted only when that history spans real time (live cameras repeat timestamps).
- **On course** if |r + ṙ·τ| < ½(1 + W_ego / W_obj).
  - W_ego = 1.8 m + 0.3 m margin.
  - W_obj priors: car 1.8 m, truck/bus 2.5 m, motorcycle 0.8 m, bicycle 0.6 m, person 0.5 m.
- **Filters:**
  - Ignore boxes whose bottom edge is above the horizon (FOE y, or the calibrated horizon row when there's no FOE).
  - Ignore classes that aren't collision-relevant.
  - Above 3°/s yaw rate, multiply the threshold by 1.5. This is a documented limitation: with no speed input, path curvature is unknown.

### 4.10 Warning FSM (PS)

Every raise also requires the track to be on course and at least 2 of the 4 TTC methods to have contributed in the last 0.5 s.

| From | To | Condition |
|---|---|---|
| NONE | WARNING | τ ≤ 2.7 s for 3 consecutive frames |
| NONE or WARNING | CRITICAL | τ + σ_τ ≤ 1.5 s (immediate) |
| WARNING | NONE | τ > 3.2 s or off course, for 10 frames; or track lost |
| CRITICAL | WARNING | τ > 2.0 s or off course, for 10 frames |
| any | NONE | track lost |

**Basis:** NHTSA's FCW confirmation test requires a warning by TTC 2.1 s (stopped lead vehicle), 2.4 s (decelerating) and 2.0 s (slower lead vehicle). 2.7 s = 2.4 s + ~0.1 s pipeline delay + 0.1 s persistence + 0.1 s margin.

The most threatening object is the on-course track with the smallest τ. Output: `Warning{track_id, level, τ, σ_τ, r_c}`.

### 4.11 Render (PS)
Overlay showing:
- box colour by warning level and every track's τ;
- the FOE heading marker;
- the collision corridor drawn from the FOE.

Output over HDMI/DP. Warning state is also exposed to software (a later CAN or GPIO output is out of scope).

## 5. Timing budget (estimates; measured in P0/P1)

| Stage | Budget per frame |
|---|---|
| T1 capture + decode | 2–4 ms |
| Preprocessing (PL) | ~3 ms |
| YOLOv9t (DPU) | ~6–8 ms, in parallel with LK |
| Vitis LK (PL) | ~12 ms |
| Horn engine (PL, P2), after LK | ~4 ms (C++ in T3 before P2) |
| T3 perception | 10–20 ms |
| T4 render | 3–6 ms |
| Capture to warning | ≈ 44 ms (target < 100 ms) |

The 33.3 ms frame period must hold for every lane; no stage is serialized across frames.

## 6. Interfaces (C++ `host/include/types.hpp`, mirrored in Python)

| Struct | Fields |
|---|---|
| `Detection` | box, class, confidence |
| `Track` | id, box, class, state [cx, cy, σ, aspect, ċx, ċy, σ̇], covariance, age, lost |
| `FlowField` | u, v (16-bit fixed point), width, height, scale, frame pair |
| `EgoMotion` | R, ω, FOE, inlier_ratio, stationary, valid |
| `BoxSums` | horn[10], flow[11], n, pyramid level |
| `TtcEstimate` | track id, τ, σ_τ, η_L, η_S, η_H, η_D, contributing-method mask |
| `Warning` | track id, level, τ, σ_τ, r_c |

Stages depend only on these structs. Swapping the Horn engine's C++ twin for HLS, or LK for learned flow, changes nothing downstream.

## 7. Repository layout

```text
collision_avoidance/fcw/  Python reference (PC), a new package: synth, tracker,
                       ego_rotation, looming, scale_search, horn, divergence, fusion,
                       collision, pipeline, render, metrics, kitti, and the CLI
                       (python -m collision_avoidance.fcw). Legacy ttc.py, tracking.py and
                       ego_motion.py stay untouched because ttc_esn imports them.
host/                  C++17 on-board app (CMake, Kria Ubuntu)
  include/types.hpp
  src/                 capture, accel (VART, XRT), tracker, ego_rotation,
                       ttc/{looming, scale_search, horn, divergence, fusion},
                       collision, render, main
  tests/               GoogleTest; golden vectors exported from Python
hls/ttc_engine/        P2 kernel: src/, tb/, script.tcl, Makefile
hls/image_derivative/  existing Sobel; fix offset, reuse in ttc_engine
hls/ports/             retired drafts, moved to legacy/
platform/              overlay: DPU B4096 + Vitis LK + ttc_engine → .xclbin
deploy/vitis_ai/       existing YOLOv9t-Hardswish flow; P3 adds PWC-DC-Net QAT
eval/                  scripts: EvTTC TTC + lead time + variance calibration, KITTI raw
                       yaw RMS, false alarms on normal-driving clips (TSTTC once obtained)
```

The on-board app is C++ because T3 needs real threads, and Python's global interpreter lock would serialize it. Python remains the reference implementation and the evaluation harness.

## 8. Testing and evaluation

| Layer | Checks | Pass condition |
|---|---|---|
| Synthetic scenes | each TTC method and ego-rotation against exact ground truth | per-method tolerance; R within 0.1° |
| Python ↔ C++ parity | each C++ stage on recorded inputs | per-stage tolerance (e.g. τ within 1%) |
| C++ ↔ HLS | engine C-simulation against the fixed-point C++ twin | bit-exact sums |
| Datasets | TTC relative error (TSTTC, EvTTC); yaw rate vs OXTS (KITTI raw); warning lead time; false alarms per hour (KITTI raw, Nexar clips) | phase exit criteria; all runs logged to W&B |
| On board | per-stage latency histograms, capture-to-warning latency, FPS, board power | 30 FPS, < 100 ms |

Ablations remove one TTC method at a time to show each method's contribution.

## 9. Phases and exit criteria

| Phase | Work | Exit criteria |
|---|---|---|
| P0 board bring-up | overlay with DPU B4096 + Vitis LK; run upstream `yolov9t_kv260.xmodel`; time each stage | per-stage latency measured; fabric fits (otherwise B3136, or LK at 256×192) |
| P1 stable baseline | C++ host app with all components of section 4 (Horn as C++ twin); Python reference updated; eval suite | 30 FPS on dash-cam clips; fused median relative TTC error ≤ 20% on EvTTC (also reported on TSTTC); yaw RMS < 1°/s on KITTI raw; ≥ 90% of approach events warned before ground-truth τ = 2.0 s; < 1 false alarm per 10 min of normal driving |
| P2 custom hardware | Horn TTC engine in HLS replaces the C++ twin | bit-exact C-sim; same warnings as P1 on the eval set; on-board speedup and T3 load reduction measured |
| P3 learned flow (stretch) | PWC-DC-Net QAT on GPU using upstream's DPU graph; warp and correlation on ARM | INT8 end-point error ≤ 0.3 px (hard ceiling 0.72 px) and divergence TTC close to the float model; then it replaces LK behind `FlowField` |
| P4 if P3 misses | choose among: mixed precision for the flow head, larger frame gap, distillation, a fused warp + correlation kernel | defined when P3 reports |

## 10. Changes from the original repository

| Area | Original | This design | Reason |
|---|---|---|---|
| Ego-motion | genetic search, 80 full-frame warps, translation removed too | essential-matrix RANSAC on LK samples, rotation removed only | forward motion is the collision signal; the search was the costliest stage |
| Flow-magnitude TTC | distance from box size, speed from flow magnitude | removed | distance cancels, leaving f / (30·|flow|) |
| Divergence TTC | Sobel derivatives, τ = 1/(div·fps) | affine fit, τ = 2kΔt / div | removes Sobel's 8× gain and the missing factor of 2 |
| Main TTC signal | optical flow | object scale change (looming, scale search, Horn) plus flow divergence | upstream measured 95% error for the flow heuristics vs 20% for expansion rate |
| Tracker | prediction read the wrong track | log-scale state with scale rate | fixes the bug; looming comes for free |
| Collision zone | fixed rectangle 140–360 × 100–300 px | FOE-relative course check | ignores objects we will pass; works at any lane position |
| Thresholds | 1.12 s in ROI, 0.56 s outside | 2.7 s warning, 1.5 s critical | the originals fire after NHTSA's 2.0–2.4 s limits |
| Detector | YOLOv9t with SiLU (fails to compile) | YOLOv9t with Hardswish | compiles to one DPU subgraph upstream |
| Learned flow | PWC-Net INT8 post-training quantization | LK baseline, QAT as stretch | INT8 error (0.95 px) exceeded the per-frame motion (0.72 px) |

## 11. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| LK noisy at night or on low-texture road | worse rotation and divergence | hold rotation when inliers < 50%; fusion down-weights divergence; scale methods don't need flow |
| Horn assumes a flat object and small motion | bias on pedestrians and very close objects | per-box pyramid level, conditioning and residual gates; fusion tolerates dropouts |
| Fabric resources | overlay doesn't fit | P0 check; B3136, 8 engine boxes, or 48-bit sums |
| No ego speed | errors in curves | widen corridor above 3°/s yaw; documented limitation |
| Wrong class width prior | corridor slightly off | affects the threshold only, not r |
| QAT misses 0.3 px | no learned flow | LK stays; P4 |

## 12. Prototype findings (2026-10-06)

The algorithm core was prototyped and tested (62 tests, Python 3.11 and 3.14) before the implementation plan was written. Changes folded into the sections above:

- Horn needs ΣEt² for its residual gate and variance: 10 Horn terms, 22 engine sums, 2,816 B per frame.
- Horn's gradients come from the frame sum (12 b), with a fixed gain of 16 removed in software.
- Ego-rotation: homography-vs-essential model selection; RANSAC inlier share as confidence.
- Divergence: RANSAC refit; de-rotation sign convention fixed to R prev → curr.
- Fusion: per-method model-error floors; re-initialisation after repeated gating.
- Warning FSM: leaving the path clears a warning the same way a long TTC does.
- Pipeline: a repeated or out-of-order timestamp falls back to the nominal frame period.
- On a synthetic head-on approach the pipeline warns at true TTC 2.73 s and goes critical at 1.27 s. Scale search is the most accurate single method (about 5%); Horn reads 10–30% fast; divergence from PC (DIS) flow reads 30–40% slow because DIS smooths across object edges.

## 13. References

- shadowPunch/Optic-Flow-Estimation PR #1 and `deploy/vitis_ai/README.md`: https://github.com/shadowPunch/Optic-Flow-Estimation/pull/1
- AMD PG338, DPUCZDX8G features: https://docs.amd.com/r/en-US/pg338-dpu/Features
- Vitis Vision Dense Pyramidal LK: https://docs.amd.com/r/2023.2-English/Vitis_Libraries/vision/api-reference.html_2_63
- Horn, Fang, Masaki, Time to Contact Relative to a Planar Surface: https://people.csail.mit.edu/bkph/articles/Time_To_Contact.pdf
- TSTTC dataset: https://ar5iv.arxiv.org/html/2309.01539
- EvTTC dataset: https://arxiv.org/html/2412.05053v3
- RAFT INT8 accuracy loss: https://arxiv.org/pdf/2208.02808
- NHTSA FCW NCAP criteria: https://www.nhtsa.gov/sites/nhtsa.gov/files/811501.pdf
