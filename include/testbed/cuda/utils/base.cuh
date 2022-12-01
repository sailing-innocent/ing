#pragma once
#ifndef TESTBED_CUDA_UTILS_BASE_H_
#define TESTBED_CUDA_UTILS_BASE_H_
__device__ float trans(float u);
__global__ void createVertices(float* positions, float time, unsigned int width, unsigned int height);

#endif // TESTBED_CUDA_BASE_H_