from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='maxmin_cuda',
    ext_modules=[
        CUDAExtension('maxmin_cuda', [
            'maxmin_cuda.cpp',
            'maxmin_cuda_kernel.cu',
            ],extra_compile_args={'cxx': [],
                                  'nvcc': ['-Xptxas=-O3,-v'
                                      #, '-maxrregcount=127'
                                      ]}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
