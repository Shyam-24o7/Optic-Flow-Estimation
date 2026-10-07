# P1 PC evaluation: FCW algorithm core

- **Plan:** `docs/superpowers/plans/2026-10-06-fcw-algorithm-core.md`, Task 13
- **Status:** all PC criteria measured. **False alarms fail badly** (117 per 10 min vs < 1); diagnosis in progress (see below).
- **Environment:** Windows 11, Python 3.11 venv, torch 2.6.0+cu124 (RTX 4060 Laptop GPU), OpenCV 4.10 DIS flow as the PC stand-in for the board's Vitis LK.

## Exit criteria (PC half of P1)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| Yaw-rate RMS vs KITTI OXTS | < 1.0 °/s | **0.681 °/s** (805 frame pairs, 3 drives) | **pass** |
| Fused median relative TTC error (EvTTC) | ≤ 20% | **11.5%** with the stale-estimate fix (12.7% after calibration, 16.2% before) | **pass** |
| Approach events warned before true TTC 2.0 s (EvTTC) | ≥ 90% | **100%** (5 of 5) | **pass** |
| False alarms on normal driving | < 1 per 10 min | **117.2 per 10 min** (352 raises in 30.0 min) | **fail** |
| Fusion `var_scale` calibration | from EvTTC | looming 0.19, scale 6.61, horn 68.13, divergence 1012.81 | applied |

## Ego-rotation detail

`python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_{0005,0014,0091}_sync`

| Drive | Frame pairs | RMS (°/s) |
|---|---|---|
| 2011_09_26_drive_0005_sync | 153 | 0.589 |
| 2011_09_26_drive_0014_sync | 313 | 0.817 |
| 2011_09_26_drive_0091_sync | 339 | 0.572 |
| **overall** | **805** | **0.681** |

Every frame pair produced a valid estimate. The low error also confirms the rotation sign convention (R maps camera coordinates prev → curr; OXTS `wu` compares directly with `rvec[1] / dt`): a flipped sign would roughly double the error on turns.

## TTC detail (EvTTC, 5 car-rear approach sequences)

`python eval/evttc_fcw.py --data data/evttc`, YOLOv9t detector, DIS flow, 512×384 with EvTTC's intrinsics scaled (non-square pixels).

| Sequence | Fused error before | Fused error after calibration | Coverage |
|---|---|---|---|
| CCRs-1-high-100 | 0.126 | 0.089 | 0.37 |
| CCRs-1-low-100 | 0.188 | 0.154 | 0.92 |
| CCRs-1-medium-100 | 0.171 | 0.137 | 0.95 |
| CCRs-2-high-100 | 0.138 | 0.114 | 0.52 |
| CCRs-2-low-100 | 0.171 | 0.148 | 0.98 |
| **all** | **0.162** | **0.127** | 0.71 |

Per method, before calibration (median relative error, coverage): looming 0.170 / 0.53, scale search 0.268 / 0.47, Horn 0.256 / 0.45, LK divergence **0.095** / 0.56.

Findings:
- On real footage divergence is the most accurate method by median, the opposite of the synthetic scenes, but its stated variance is far too small on its bad frames (calibration factor about 1000). Its typical accuracy is good, its outliers are large.
- The calibrated factors replace the defaults of 1.0 in `FusionConfig.var_scale`; all 66 FCW tests still pass with them.
- Coverage is lowest on the "high" (fast-approach) sequences: the target is often too small or too close for a confident estimate at the start and end.

## False alarms (30 min of UK residential driving)

`python eval/false_alarms.py --videos data/normal_driving --fx 200`: 10 front-camera clips from the MIT-licensed `aap9002/UK-Road-DashCam` dataset (4K, 30 FPS, no calibration published; fx = 200 px at 512×384 assumed for a wide dash cam).

| Clip | Raises |
|---|---|
| 241220_125301_002 | 35 |
| 241220_125601_003 | 32 |
| 241220_125901_004 | 27 |
| 241220_130201_005 | 36 |
| 241220_130501_006 | 52 |
| 241220_130801_007 | 31 |
| 241220_131102_008 | 38 |
| 241220_131402_009 | 35 |
| 241220_131702_010 | 12 |
| 241220_132302_012 | 54 |
| **total** | **352 in 30.0 min = 117.2 per 10 min** |

Diagnosis on clip 003 (first 1,500 frames, 19 raises):
- Every raise is a car (18) or truck (1) and goes straight to CRITICAL. The TTC itself is plausible: these are oncoming cars and parked cars on narrow residential streets, closing fast in the image, and all methods agree.
- The **course check** is what fails. The heading (FOE x) jumps between 210 and 416 px on a 512 px frame, so the lateral-offset rate ṙ is noise and `r + ṙ·TTC` lands near 0 for cars that are currently 1.5–10 widths to the side (e.g. r = 4.30, r_contact = 0.52).
- Two real-footage bugs found on the way were fixed in Python and C++ (lost tracks holding warnings; stale predictions driving TTC to 0). They cut critical frames on the first clip from 88 to 67 but did not touch the course-check problem.

## To finish this report

1. Fix the course check's heading instability and re-run `eval/false_alarms.py`; re-check EvTTC lead time.
