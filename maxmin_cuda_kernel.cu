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

__global__ void maxmin_tensor(float* dst,float* src,int total,float* max,float* min_src) {
    int wch_max = blockIdx.x;
    float maxV = max_src[wch_max];
    float minV = min_src[wch_max];
    int tid =  blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > total) return;
    int offset = wch_max * total;
    for(;tid < total ;tid += blockDim.x * gridDim.x) {
        dst[offset + tid] = min(maxV,max(minV,src[offset + tid]));
    }

}

__global__ void maxmin_single(float* dst,float* src,int total,float* max_src,float* min_src) {
    float maxV = max_src[0];
    float minV = min_src[0];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > total) return;
    for(;tid < total ;tid += blockDim.x * gridDim.x) {
        dst[tid] = min(maxV,max(minV,src[tid]));
    }
}

torch::Tensor maxmin_cuda_forward(
    torch::Tensor input,
    torch::Tensor max,
    torch::Tensor min) {
    const auto batch_size = input.size(0);
    const auto C  =input.size(1);
    const auto H  =input.size(2);
    const auto W  =input.size(3);

    auto dtype = input.type();
    auto dev = input.device();
    auto output = torch::empty({batch_size, C, H, W}, dtype);

    const auto maxSize = max.size(0);

    if (maxSize == 1) {
        const auto total = batch_size * C * H * W;
        dim3 grid(32);
        dim3 block(256);
        maxmin_single<<<grid,block>>>((float*)output.data_ptr(),(float*)input.data_ptr(),total,(float*)max.data_ptr(),(float*)min.data_ptr());
    } else {
        dim3 grid(batch_size);
        dim3 block(256);
        const auto total = C * H * W;
        maxmin_tensor<<<grid,block>>>((float*)output.data_ptr(),(float*)input.data_ptr(),total,(float*)max.data_ptr(),(float*)min.data_ptr());
    }
    return output;
}
