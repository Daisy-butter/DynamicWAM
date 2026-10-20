# Step20000 camera-backlog evaluation restart, 2026-10-20

Parent cb89c75831d97c9ba7f917e7397b01765ac662f8. Failed formal03 is preserved: POB episode16 peaked at332.788ms of physical backlog during synchronous camera submission, above250ms. Suffixes were timely; this was not a model deadline or new driver fault.

The user requests a complete35x20 before further architecture changes and permits the earlier hardware concession. New explicit joint-v3 fixed0.25 protocol retains exact20k weights, physical dt.004/action.16, H4/R4/K10, source2+8, refresh16, maxage31/4.96, normalization, all35tasks and frozen700scenes. Default1x stays available; no adaptive speed, timeout relaxation, task omission, automatic retries or1x claim. Timing distribution is different from1x and results must be reported separately.

Runtime now reports true global-clock validity and phase/query/timestamps for the worst backlog, without moving observations or committed deadlines. Full report/video validation precedes the finished-cohort barrier; invalid report/metrics/nativeMP4 remain saved. Pipeline binds fixed rate/protocol and requires scaled clock-valid plus realtime-valid=false. Scaled stage budgets6h/40h remain below finite48h kernel broker lifetime; no permanent keepalive.

CPU validation:39 controller/v3/clock tests,10 finished-episode tests,27 pipeline tests passed. Unchanged graph/model still receives fresh actual20-case Graph/background acceptance bound to the new source/config, then8GPU8tasks x2full episodes includingPOB/PCP,16decoded MP4s and clean device-FD release before all700 formal launch. These GPU checks and formal result are pending at this source freeze; do not read this document as acceptance. Large artifacts use the existing verified .100 SSD; .50 training is untouched.

Outputs: ../outputs/joint20k_35x20_04_scaled025; controller ../repair04/status.json. Preserve all earlier attempts and require source/config freeze during active jobs. No next architecture change until evaluation completes.
