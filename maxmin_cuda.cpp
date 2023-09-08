#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor own_max_min_cuda(
    torch::Tensor input,
    torch::Tensor min,
    torch::Tensor max
);




// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.


torch::Tensor own_max_min(
    torch::Tensor input,
    torch::Tensor minT,
    torch::Tensor maxT) {
  return own_max_min_cuda(input, minT, maxT);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("own_max_min",  &own_max_min,  "own_max_min (CUDA)");
}



TORCH_LIBRARY(own_max_min, m) {
    m.def("own_max_min", own_max_min);
}
