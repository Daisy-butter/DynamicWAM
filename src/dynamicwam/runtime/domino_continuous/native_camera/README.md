# SAPIEN Color readback bridge

This narrow extension calls SAPIEN 3.0.0b1 `SapienRenderCameraComponent::getImage("Color")` with the Python GIL released, then copies the completed HWC RGBA float32 image into a NumPy-owned allocation. It neither updates render state nor submits a camera render. A Python owner reference keeps the camera alive throughout the native call.

The installed Python `get_picture` binding holds the GIL across both render completion and CPU download. Merely wrapping it in a Python thread, or waiting through `get_picture_cuda` first, does not remove that blocking path.

## Use and concurrency contract

```python
import sapien  # bootstrap the wheel native dependencies (including libsvulkan2) first
from dynamicwam.runtime.domino_continuous import _sapien_color_reader
rgba = _sapien_color_reader.read_color(camera)
```

The owner thread freezes the observation timestamp, robot state and camera metadata, calls `update_render` and `take_picture`, then delegates only this readback call. While readback is pending it may advance the CPU PhysX scene and the existing dynamic-object updater. Do not call `update_render`, submit another camera frame, modify render resources, or destroy the camera/scene until every readback future has returned. The returned array remains valid after the next camera capture.

This is a narrow concurrency contract inferred from the installed headers and official SAPIEN 3.0.0b1 source. It is not a claim that arbitrary SAPIEN operations are thread-safe. In particular, moving all of `env.get_obs` into a worker would mix observation times and potentially mutate render state concurrently.

## Build and ABI checks

Run `native_camera/build_color_reader.py` with the exact DOMINO Python, `--project /home/wuruihan/DynamicWAM`, and the provided `--pybind-archive` path. Add `--prepare-toolchain` for the first build. The script downloads official Ubuntu GCC10 packages, verifies the apt metadata hashes, and unpacks them only under `external/native_camera_deps`. It does not install system packages or replace SAPIEN.

The build requires the observed installed pybind internals key `__pybind11_internals_v4_gcc_libstdcpp_cxxabi1014__` and naturally matching GCC10 compiler ABI 1014. The mangled exported `getImage` symbol verifies `_GLIBCXX_USE_CXX11_ABI=0`. The extension uses the official smart-holder headers and normal registered C++ type casting; no pointer extraction or duplicate camera type registration is involved.

SAPIEN's historical build used a floating `smart_holder` branch. Its wheel does not record the exact pybind commit. The pinned dependency is `cfe73560a252a4a0362f0bfe883c31c67770e266`, the latest first-parent commit of the official archived branch preceding the 2024-05-22 SAPIEN 3.0.0b1 release. This limitation is recorded in `_sapien_color_reader.build.json` rather than claiming an exact wheel-build source match.

The build checks module import, identity of the registered camera class, ordinary typed casting, twenty native exceptions from an unattached camera with GIL recovery, and wrong-type rejection. These checks create no GPU scene. The final artifact and library hashes, arguments, dependency source and CPU results are recorded in `_sapien_color_reader.build.json`. Passing these checks still requires actual same-frame image equivalence and concurrent physics validation before formal evaluation.
