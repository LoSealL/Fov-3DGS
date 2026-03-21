from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the CUDA extension
cuda_ext = CUDAExtension(
    name="_C",  # Importable module name
    sources=["csrc/kernel.cu"],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "-Xcompiler", "/Zc:preprocessor"],
    },
)

setup(
    name="testCudaExt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[cuda_ext],
    cmdclass={"build_ext": BuildExtension},
)
