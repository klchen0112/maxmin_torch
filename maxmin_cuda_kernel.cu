#include <torch/extension.h>
#include <thrust/device_vector.h>

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cooperative_groups.h>
//#include <cuda/barrier>
#include <vector>
#include <vector_types.h>

__global__ void maxmin_tensor(float* dst,const float* src,int total,const float* min_src,const float* max_src) {
    int wch_max = blockIdx.x;
    float maxV = max_src[wch_max];
    float minV = min_src[wch_max];
    int tid =  threadIdx.x;
    if (tid >= total) return;
    const int offset = wch_max * total;
    for(;tid < total ;tid += gridDim.x) {
        dst[offset + tid] = min(maxV,max(minV,src[offset + tid]));
    }

}

__global__ void maxmin_single(float* dst,const float* src,const int total,const float* min_src,const float* max_src) {
    float maxV = max_src[0];
    float minV = min_src[0];
    int tid = threadIdx.x;
    if (tid > total) return;
    for(;tid < total ;tid += blockDim.x) {
        dst[tid] = min(maxV,max(minV,src[tid]));
    }
}

torch::Tensor own_max_min_cuda(
    torch::Tensor input,

    torch::Tensor min,
    torch::Tensor max) {
    const auto batch_size = input.size(0);
    const auto elements  = input.numel() / batch_size;

    auto dtype = input.type();
    auto dev = input.device();
    auto output = input.new_empty(input.sizes()).to(dtype);

    const auto maxSize = max.size(0);

    if (maxSize == 1) {
        const auto total = batch_size * elements;
        dim3 grid(1);
        dim3 block(256);
        maxmin_single<<<grid,block>>>((float*)output.data_ptr(),(float*)input.data_ptr(),total,(float*)min.data_ptr(),(float*)max.data_ptr());
    } else {
        dim3 grid(batch_size);
        dim3 block(256);
        const auto total = elements;
        maxmin_tensor<<<grid,block>>>((float*)output.data_ptr(),(float*)input.data_ptr(),total,(float*)min.data_ptr(),(float*)max.data_ptr());
    }
    return output;
}
