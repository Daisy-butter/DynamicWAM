# Stage3 step20000 independent evaluation on .100

Protected training: .50 worktree joint-motionkv-20261015, HEAD b4da95704b647f128e2b6e1374b5e28c74607fad; no training/GPU/process changes. Evaluation derives from normalization fix70cd2b0 in a separate .50 worktree. The user authorizes publishing this feature branch to Daisy-butter and a35x20 continuous RTC evaluation on .100.

20k full SHA9ca43399ba0c40e529744682a0f52ebf794db64e8252746d6187ac358a30f5b1; fast SHA07ead62f0357c0ba86ff2ebca26df381cb0ca638174a2e29266df6a5577e8b57; exact exported backbone SHAe133943819710fb0bd13d3b424da243ee2b479dae55dbbb8b0e2713fc942f9cc. Full709/fast370 tensors finite and strict CPU reload passes, source-time module retained. Destination audits must rebind only paths after SHA validation.

Inference repairs: FP32 normalization before model precision cast; strict source physical-time encoder schema loading; worker initialization failure cleanup; explicit fast/slow episode drain before next cohort setup. Existing H4/R4/K10/refresh16, age31, action0.16s and1x continuous wall clock are retained. No few-step override, clock slowdown, deadline extension or new trained modules.

Exact DOMINO6c5a00db source is reproducible with manifests/external_assets_stage3_20k_20261019.json and its versioned patch. Original700 prepared scenes including POB/PCP retain immutable hashes; a separate receipt verifies scene-relevant configuration/source compatibility across model and path changes. Native eval_result reports, actual10fps head-camera MP4 and physical-time sidecars are retained.

Target root: /SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019. Independent copied Python runtime disables user site packages; system/home packages must not override it. Asset/runtime binaries and weights are not committed. Kernel-log broker only permits fixed read-only dmesg for this UID, with bounded lifetime and explicit shutdown, and never modifies driver/GPU state.

Acceptance order: local source/asset/checkpoint verification -> fresh guarded telemetry -> actual20k Graph parity/background slow concurrency -> eight tasks x two full native episodes plus MP4 decode and clean guard release -> formal35x20. No automatic retry; failures remain recorded. Launch alone is not successful evaluation.
