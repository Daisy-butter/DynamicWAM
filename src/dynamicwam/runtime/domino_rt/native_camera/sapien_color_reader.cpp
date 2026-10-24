// Narrow SAPIEN 3.0.0b1 bridge: wait/download one already-submitted Color frame.
// Rendering, scene updates, camera mutation and destruction stay with the owner.
#include <pybind11/numpy.h>
#include <pybind11/smart_holder.h>
#include <sapien/sapien_renderer/camera_component.h>
#include <cstring>
#include <stdexcept>
#include <string>
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

PYBIND11_MODULE(_sapien_color_reader, module) {
  py::module_::import("sapien.pysapien");
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
  module.def("read_color", &read_color, py::arg("camera"),
      "Copy one submitted Color frame while releasing the GIL for wait and download. "
      "Caller must prevent render updates, new captures and scene/camera destruction until return.");
}
