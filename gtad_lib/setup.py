from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='Align1D',
    version="2.2.0",
    author="Frost Mengmeng Xu",
    author_email="xu.frost@gmail.com",
    description="A small package for 1d aligment in cuda",
    long_description="I will write a longer description here :)",
    long_description_content_type="text/markdown",
    url="https://github.com/Frostinassiky/G-TAD",
    ext_modules=[
        CppExtension(
            name = 'Align1D', 
            sources = [
              'Align1D_cuda.cpp', 
              'Align1D_cuda_kernal.cu',
            ],
            extra_compile_args={'cxx': [],
              'nvcc': ['--expt-relaxed-constexpr']}
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
