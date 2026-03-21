#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(float *a, float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
  TORCH_CHECK(b.device().is_cuda(), "b must be on CUDA");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same size");

  auto out = torch::empty_like(a);
  int n = a.numel();
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  add_kernel<<<grid_size, block_size>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add_cuda);
}