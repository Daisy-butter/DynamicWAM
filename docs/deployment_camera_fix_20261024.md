# Camera follow-up after deployment02

Deployment02 completed full native model initialization and the initial slow/fast
queries. It failed because a query at physical time 0.360 used the latest
available exposure at 0.060. Its exposure-to-availability interval was 0.124 s;
the recorded readback alone was 0.032 s. The missing owner/dispatch timing means
the exact latency source is not established. Original failure remains preserved.

The candidate now uses the previously implemented native four-camera RGB batch
bridge, with the same cameras, rendering settings and pixel conversion. Three
fixed captures warm the camera before activation; only the final snapshot enters
model history/video. Active physics remains 1x. Due queries with stale pixels
wait for fresh completed pixels while the existing executable queue continues;
their lease is created at actual query time on the unchanged floor-based grid.
Protect2, deadline, TTL, underflow and 0.060001 s observation-age bounds remain.
Capture phase timings and remaining committed-prefix budget are recorded.

## Compile only on .92

Use the isolated candidate; do not replace the home repository's bridge.
Read-only dependency audit is `native_camera_dependency_audit92.json` in the local
work folder: installed pybind internals1014, old std::string ABI0 and GCC10.5
match. The archive SHA256 is
`efdd1f05344fc3398b5d76c84e273295182fb875732afec37bb326c6a7b1b863`.

```bash
ROOT=/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-joint-v3-20261024
"$ROOT/external/robotwin/bin/python" "$ROOT/src/dynamicwam/runtime/domino_rt/native_camera/build_color_reader.py" \
  --project "$ROOT" \
  --pybind-archive /home/wuruihan/DynamicWAM/external/native_camera_deps/pybind11-smart-holder-cfe7356.tar.gz \
  --compile-only
```

Do not pass `--prepare-toolchain`. `--compile-only` checks installed metadata,
ELF symbols, compiler macros and headers, then compiles/hashes; it does not import
SAPIEN, load the output extension, call nvidia-smi or create a GPU context.
Rebind the newly compiled `.so` SHA256 in `deployment_artifact_sha256` and freeze
source/config again. The `.build.json` explicitly marks import validation absent.

Run CPU tests including `test_dualrate_camera_scheduling.py`. Native numerical
pixel parity runs only after the next parent-guarded deployment imports the
extension, saving `native_rgb_conversion.json`. Real camera timing, new-reference
consumption, valid MP4 and clean exit are still pending. This is not formal
training readiness or a successful deployment acceptance.
