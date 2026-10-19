# RTC service latency and prefix consistency

The prior formal run stopped at POB episode0/origin84. Capture took371.900ms, leaving264.77ms until H4. Dispatch occurred before catch-up; the legacy submit_wall_s timestamp was after catch-up. Prior formal requests had dispatch-to-model-start up to139ms while median model compute remained near90ms. These observations motivate transport/service fixes; they do not prove a single exclusive root cause.

- Use native held drive targets for bootstrap committed prefix. Keep measured qpos as proprioception and interpolation anchor.
- Anonymous shared request/result mailboxes eliminate Queue feeder serialization after dispatch. Full packet snapshots are unchanged, one request in flight, overflow rejected, stop never overwrites an outstanding request.
- Fix child CPU pools at Torch1/1 and OpenCV1.
- Distinct H4 phases replace refresh16/8 spacing, which aliased four peers per fast phase. Offsets precede each independent episode clock.
- Complete camera and IPC timestamp partitions distinguish reader/GIL/service delay from rendering and compute.

H4/R4/K10, refresh16, action0.16s, physics0.004s, 1x continuous time, strict deadline failure, checkpoint and all35 tasks are unchanged. No inference-only claim of on-policy training coverage. CPU validation covers transport ownership/lifecycle, held-prefix identity, phase uniqueness and timing accounting. Actual latency improvement requires GPU closed-loop acceptance; CPU and Graph parity alone are insufficient.
