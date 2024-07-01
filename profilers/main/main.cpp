#include <torch/extension.h>
void square(const torch::Tensor& input, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("square", torch::wrap_pybind_function(square), "square");
}