#include <torch/extension.h>

void fuse_softmax(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& temperature);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fuse_softmax",
        &fuse_softmax,
        "Fusion of div, softmax, log Cuda kernel");
}
