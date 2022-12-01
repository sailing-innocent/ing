#pragma once
#ifndef TESTBED_CUDA_UTILS_TRACER_H_
#define TESTBED_CUDA_UTILS_TRACER_H_
__device__ float trans1(float u);
__global__ void createVertices1(float* positions, float time, unsigned int width, unsigned int height);

#endif // TESTBED_CUDA_BASE_H_