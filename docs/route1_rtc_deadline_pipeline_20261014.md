# RTC deadline deployment repair

The failed formal02 request at place_object_basket action20/origin16 consumed359.879ms in capture and111.803ms before submission. About168ms remained in the fixed640ms H4 window, below the preceding slow-model248-254ms. No new NVLink/IOVAS; all owned workers/guards closed.

Changes: reuse exact optical-flow pixel pairs by immutable frame IDs (features/timestamps rebuilt); bound/reset cache; explicit one OpenCV thread; phased cohorts prevent another episode setup/warmup/teardown overlapping a live episode. Add capture stage timings. No change to trained model, clean-prefix semantics, H4/R4, nominal160ms spacing, K10, slow refresh16, max cache age12, deadline, video format or frozen700scenes.

CPU actual Farneback array-equality tests and controller tests pass. Standalone cache benchmark15queries: p50 27.96ms uncached vs5.70ms cached; this does not establish end-to-end speedup.8GPU validation must cover previous failing first-eight task set,2episodes each and64action limit; then formal35x20 with original scene hashes. Preserve former attempts and fail closed without automatic retries.
