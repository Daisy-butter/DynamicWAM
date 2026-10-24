// Narrow SAPIEN 3.0.0b1 bridge: wait/download one already-submitted Color frame.
// Rendering, scene updates, camera mutation and destruction stay with the owner.
#include <pybind11/numpy.h>
#include <pybind11/smart_holder.h>
#include <sapien/sapien_renderer/camera_component.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using Camera = sapien::sapien_renderer::SapienRenderCameraComponent;

static py::array_t<float> read_color(py::object const &owner) {
  // A normal registered-type cast, with the Python argument owning the camera
  // throughout the native call. No pointer extraction or duplicate registration.
  Camera &camera = owner.cast<Camera &>();
  auto h = static_cast<py::ssize_t>(camera.getHeight());
  auto w = static_cast<py::ssize_t>(camera.getWidth());
  if (h <= 0 || w <= 0 || h > 16384 || w > 16384) {
    throw std::runtime_error("invalid camera dimensions; refusing image read");
  }
  py::array_t<float> result(std::vector<py::ssize_t>{h, w, 4});
  float *destination = result.mutable_data();
  {
    py::gil_scoped_release release;
    // getImage waits for the submitted render and performs the complete CPU
    // download. Moving only the render fence would leave download blocking.
    auto image = camera.getImage("Color");
    if (image.type != "f4" && image.type != "<f4" && image.type != "=f4") {
      throw std::runtime_error("Color image is not native little-endian float32");
    }
    const std::vector<int> shape{static_cast<int>(h), static_cast<int>(w), 4};
    const std::vector<int> strides{static_cast<int>(w * 16), 16, 4};
    if (image.shape != shape || image.strides != strides || image.ptr == nullptr) {
      throw std::runtime_error("Color image shape/strides differ from packed HWC RGBA");
    }
    // Copy before releasing camera ownership. The returned NumPy allocation is
    // independent of the camera's reusable image buffer and next capture.
    std::memcpy(destination, image.ptr, static_cast<size_t>(h * w * 4) * sizeof(float));
  }
  return result;
}

// Preserve NumPy float32 multiply -> clip -> uint8 truncation exactly.
// Alpha is not returned, but nonfinite alpha must still reject the frame.
static void convert_rgb(float const *source, uint8_t *destination, size_t pixels) {
  for (size_t pixel = 0; pixel < pixels; ++pixel) {
    for (size_t channel = 0; channel < 4; ++channel) {
      float value = source[4 * pixel + channel];
      if (!std::isfinite(value)) {
        throw std::runtime_error("native Color reader returned nonfinite Color values");
      }
      if (channel < 3) {
        float scaled = value * 255.0f;
        scaled = scaled < 0.0f ? 0.0f : (scaled > 255.0f ? 255.0f : scaled);
        destination[3 * pixel + channel] = static_cast<uint8_t>(scaled);
      }
    }
  }
}

// Expose the same conversion to a GPU-free numerical equivalence test.
static py::array_t<uint8_t> rgba_to_rgb_u8(
    py::array_t<float, py::array::c_style> const &rgba) {
  if (rgba.ndim() != 3 || rgba.shape(2) != 4 || rgba.shape(0) < 1 || rgba.shape(1) < 1) {
    throw std::runtime_error("expected nonempty float32 HWC4");
  }
  auto h = rgba.shape(0), w = rgba.shape(1);
  py::array_t<uint8_t> result(std::vector<py::ssize_t>{h, w, 3});
  auto *destination = result.mutable_data();
  auto const *source = rgba.data();
  {
    py::gil_scoped_release release;
    convert_rgb(source, destination, static_cast<size_t>(h * w));
  }
  return result;
}

static py::tuple read_rgb_batch(py::sequence const &owners) {
  if (py::len(owners) < 1) throw std::runtime_error("camera batch must not be empty");
  std::vector<Camera *> cameras;
  std::vector<py::array_t<uint8_t>> outputs;
  std::vector<py::ssize_t> heights, widths;
  std::vector<double> durations;
  // owners retains strong references until every download/conversion finishes.
  for (auto owner : owners) {
    Camera &camera = py::cast<Camera &>(owner);
    auto h = static_cast<py::ssize_t>(camera.getHeight());
    auto w = static_cast<py::ssize_t>(camera.getWidth());
    if (h <= 0 || w <= 0 || h > 16384 || w > 16384) {
      throw std::runtime_error("invalid camera dimensions; refusing image read");
    }
    cameras.push_back(&camera); heights.push_back(h); widths.push_back(w);
    outputs.emplace_back(std::vector<py::ssize_t>{h, w, 3});
  }
  durations.resize(cameras.size());
  // Acquire all NumPy pointers before releasing the GIL; native loop below
  // contains no Python API operations and reacquires only once for the batch.
  std::vector<uint8_t *> destinations;
  for (auto &output : outputs) destinations.push_back(output.mutable_data());
  {
    py::gil_scoped_release release;
    for (size_t index = 0; index < cameras.size(); ++index) {
      auto started = std::chrono::steady_clock::now();
      auto image = cameras[index]->getImage("Color");
      if (image.type != "f4" && image.type != "<f4" && image.type != "=f4") {
        throw std::runtime_error("Color image is not native little-endian float32");
      }
      auto h = heights[index], w = widths[index];
      const std::vector<int> shape{static_cast<int>(h), static_cast<int>(w), 4};
      const std::vector<int> strides{static_cast<int>(w * 16), 16, 4};
      if (image.shape != shape || image.strides != strides || image.ptr == nullptr) {
        throw std::runtime_error("Color image shape/strides differ from packed HWC RGBA");
      }
      convert_rgb(static_cast<float const *>(image.ptr), destinations[index],
                  static_cast<size_t>(h * w));
      durations[index] = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
    }
  }
  py::list images, elapsed;
  for (size_t index = 0; index < outputs.size(); ++index) {
    images.append(outputs[index]); elapsed.append(durations[index]);
  }
  return py::make_tuple(images, elapsed);
}

static void take_picture(py::object const &owner) {
  // Same call as Camera.take_picture. SAPIEN's binding keeps the GIL while
  // waitForRender blocks, which freezes every Python inference worker.
  Camera &camera = owner.cast<Camera &>();
  py::gil_scoped_release release;
  camera.takePicture();
}

// libsapien PhysxSystemCpu::step(). The physx headers do not compile with this
// bridge's C++20 mode, so the call uses the installed symbol directly.
extern void physx_cpu_step(void *self) __asm__("_ZN6sapien5physx14PhysxSystemCpu4stepEv");

static void step_cpu_physx(py::object const &owner) {
  py::object expected = py::module_::import("sapien.pysapien.physx").attr("PhysxCpuSystem");
  if (!py::isinstance(owner, expected)) {
    throw std::invalid_argument("step_cpu_physx requires the installed PhysxCpuSystem");
  }
  auto *inst = reinterpret_cast<py::detail::instance *>(owner.ptr());
  if (inst == nullptr) {
    throw std::runtime_error("CPU PhysX object is not a pybind instance");
  }
  // SAPIEN registers PhysX with a holder larger than shared_ptr, so the
  // instance uses the nonsimple layout. The first value is the system itself.
  void *self = inst->get_value_and_holder().value_ptr();
  if (self == nullptr) {
    throw std::runtime_error("CPU PhysX instance has no registered C++ pointer");
  }
  // Scene.step is this call. SAPIEN's binding keeps the GIL for simulate/fetch,
  // so a tight catch-up loop starves the Python inference workers.
  py::gil_scoped_release release;
  physx_cpu_step(self);
}

static void sleep_releasing_gil(double seconds) {
  if (!(seconds > 0.0) || seconds > 2.0) {
    throw std::invalid_argument("sleep_releasing_gil expects 0 < seconds <= 2");
  }
  py::gil_scoped_release release;
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

PYBIND11_MODULE(_sapien_color_reader, module) {
  py::module_::import("sapien.pysapien");
  py::module_::import("sapien.pysapien.physx");
  auto *registered = py::detail::get_type_info(typeid(Camera), false);
  if (registered == nullptr) {
    throw py::import_error("SAPIEN camera type is absent from matching pybind registry; ABI mismatch");
  }
  py::object expected = py::module_::import("sapien.pysapien.render").attr("RenderCameraComponent");
  if (reinterpret_cast<PyObject *>(registered->type) != expected.ptr()) {
    throw py::import_error("SAPIEN camera registration does not match the installed class");
  }
  module.attr("pybind_internals_id") = PYBIND11_INTERNALS_ID;
  module.attr("pybind_source_commit") = "cfe73560a252a4a0362f0bfe883c31c67770e266";
  module.attr("cpp_string_abi") = _GLIBCXX_USE_CXX11_ABI;
  module.attr("compiler_cxxabi") = __GXX_ABI_VERSION;
  module.attr("rgb_batch_contract") = "float32_rgba_to_u8_rgb_exact_v1";
  module.def("rgba_to_rgb_u8", &rgba_to_rgb_u8, py::arg("rgba").noconvert(),
             "GPU-free exact Color conversion, including alpha finite validation.");
  module.def("read_rgb_batch", &read_rgb_batch, py::arg("cameras"),
             "Download all submitted cameras and convert RGB under one GIL release; "
             "returns owning images and per-camera download/conversion seconds.");
  module.def("read_color", &read_color, py::arg("camera"),
      "Copy one submitted Color frame while releasing the GIL for wait and download. "
      "Caller must prevent render updates, new captures and scene/camera destruction until return.");
  module.def("take_picture", &take_picture, py::arg("camera"),
      "Submit one camera render while releasing the GIL across waitForRender.");
  module.def("step_cpu_physx", &step_cpu_physx, py::arg("system"),
      "Advance one CPU PhysX step while releasing the GIL.");
  module.def("sleep_releasing_gil", &sleep_releasing_gil, py::arg("seconds"),
      "CPU witness that this module releases the GIL for the whole wait.");
}
