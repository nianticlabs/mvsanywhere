#include <torch/extension.h>
#include "marching_cubes.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("marching_cubes_", &MarchingCubes);
}
