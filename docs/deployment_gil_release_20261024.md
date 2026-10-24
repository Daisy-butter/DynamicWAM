# Owner GIL release for DualRate deployment

Smoke07 satisfied the 0.24 s image-age ceiling and still underran at 2.880 s.
Fast query 1 started at physics 0.360 s with protected_until 0.720 s. Its
actuator projection is the last event, after the capture submitted at exposure
2.880 s. The fast slot stayed occupied, so later queries were never dispatched.
Uncontended bootstrap sampling of the same plan finished in 0.302 s.

SAPIEN 3.0.0b1 binds `take_picture` and `System::step` without releasing the
GIL. `takePicture` waits on the previous frame, and `PhysxSystemCpu::step` runs
simulate plus fetchResults. Fast and slow workers are Python threads, so they
cannot run while the owner thread is inside those calls. Bootstrap inference
only finishes quickly because the owner sleeps and releases the GIL.

This change keeps the DualRate model, periods, age ceiling, and backlog gate.
The native bridge releases the GIL around each camera `takePicture` and around
the installed `PhysxSystemCpu::step` symbol. The owner polls again after a
blocking capture and before `action_at`, so a plan that finishes during render
can be committed on that tick. A deadline miss still cancels the lease only
through the existing timing rules; this note does not add a second in-flight
fast job.
