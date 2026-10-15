# Route1 fixed-rate continuous evaluation

Parent b2ec6d2; branch feat/route1-scaled-continuous-eval-20261015.

Training uses uniform clean expert prefix d=0..4 (multirate_trainer.py:731), suffix-only flow loss (rtc_prefix.py:70), cache ages0/4/8/12 (paired_index.py:43), real sim_time_seconds for motion features. No training model of wall camera/renderer latency. Preserve full step12000 weights, K10/H4/R4, action0.16s, refresh16 and simulated dynamics.

PCP prior episode14 failed at sim7.384/wall7.6376, backlog0.2536>0.25 with no pending inference/camera. Mean measured physics step1.691ms, max15.091ms vs4ms dt. Fixed rate0.25 gives16ms wall per physics tick and2.56s wall per640ms H4. This is a declared hardware concession and different protocol, not a repaired1x result. World keeps advancing on the declared clock during inference/capture; no adaptive slowing or latency removal. Actual model and dispatch latency remain real wall values. Old post-drain submit timestamp is explicitly labeled legacy, new dispatch timestamp precedes worker.submit.

Initial verifier: meaningful fake-clock tests, independent saved-report validation, existing fences. Actual8tasks×2fullnative episodes includesPOB/PCP; all16videos decoded and guard closure verified before pipeline advances to35×20 on all700preserved scenes. All8cards concurrently, no excluded tasks. No automatic retry. Large results stay on /SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/motionkv_scaled_r025_20261015. Rate and protocol recorded in every task/suite/report; never mix with actual1x partial scores. A baseline with identical rate/scenes is required to attribute performance improvement to multirate.
