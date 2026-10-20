# Stage3 step30000 evaluation replacement

User requested replacement on2026-10-20. Inference/model/controller source is unchanged from2d4fdf150c2f70334e7cdd510d8a44070d4838d9. The evaluation YAML changes only backbone path and output label.

Checkpoint: /SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019/checkpoint30k/multirate_step_30000.pt
SHA256: 39fc723af910b857cd6f50665da570778734619c64c71bcaf69b8d900224efa9
Full/fast strict CPU reload, finite weights, source-time backbone extraction, full-data/normalization and runtime contract equality to20k passed. Source and destination transfer hashes match. No CUDA was initialized by the CPU audit.

Old20k formal saved16reports before the user-requested interruption; guard workers/FDs all closed and driver_exit_clean=true. Its results remain under outputs/joint20k_35x20_04_scaled025 and are not included in30k scores. Old failed attempts remain untouched.

New run: outputs/joint30k_35x20_01_scaled025. Fixedsimulation_rate0.25, realtime_valid=false, unchangedH4/R4/K10/refresh16/age31 and35x20scene set. New preflight, actual30k Graph/background and8x2full-episode smoke must pass before formal700. This document records preparation, not a claim that formal started or completed. .50 training was not modified or stopped.
