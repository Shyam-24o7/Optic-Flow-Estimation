# P1 PC evaluation: FCW algorithm core

- **Plan:** `docs/superpowers/plans/2026-10-06-fcw-algorithm-core.md`, Task 13
- **Status:** partial. Ego-rotation measured; TTC accuracy, warning lead time, calibration and false alarms still to run.
- **Environment:** Windows 11, Python 3.11 venv, torch 2.6.0+cu124 (RTX 4060 Laptop GPU), OpenCV 4.10 DIS flow as the PC stand-in for the board's Vitis LK.

## Exit criteria (PC half of P1)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| Yaw-rate RMS vs KITTI OXTS | < 1.0 °/s | **0.681 °/s** (805 frame pairs, 3 drives) | **pass** |
| Fused median relative TTC error (EvTTC) | ≤ 20% | not run | blocked: EvTTC videos refused by Google Drive (quota) |
| Approach events warned before true TTC 2.0 s (EvTTC) | ≥ 90% | not run | same |
| False alarms on normal driving | < 1 per 10 min | not run | needs dash-cam clips in `data/normal_driving/` |
| Fusion `var_scale` calibration | from EvTTC | not run | defaults (1.0) remain |

## Ego-rotation detail

`python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_{0005,0014,0091}_sync`

| Drive | Frame pairs | RMS (°/s) |
|---|---|---|
| 2011_09_26_drive_0005_sync | 153 | 0.589 |
| 2011_09_26_drive_0014_sync | 313 | 0.817 |
| 2011_09_26_drive_0091_sync | 339 | 0.572 |
| **overall** | **805** | **0.681** |

Every frame pair produced a valid estimate. The low error also confirms the rotation sign convention (R maps camera coordinates prev → curr; OXTS `wu` compares directly with `rvec[1] / dt`): a flipped sign would roughly double the error on turns.

## To finish this report

1. `python -m ttc_esn.evttc --root data/evttc` once Google Drive's quota resets, then `python eval/evttc_fcw.py --data data/evttc`.
2. Put 30+ minutes of normal-driving dash-cam footage in `data/normal_driving/`, then `python eval/false_alarms.py --videos data/normal_driving --fx <fx>`.
3. Copy `var_scale` from step 1 into `FusionConfig` (plan Task 13, Step 3) and rerun.
