#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor maxmin_cuda_forward(
    torch::Tensor input,
    torch::Tensor max,
    torch::Tensor min
);




// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.


torch::Tensor maxmin_forward(
    torch::Tensor input,
    torch::Tensor maxT,
    torch::Tensor minT) {
  return maxmin_cuda_forward(input, maxT, minT);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &maxmin_forward,  "maxmin forward (CUDA)");
}
