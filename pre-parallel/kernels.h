/*
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();
__global__ void kernelForward(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* d_train_data, int n);

// __global__ void kernelLoss(float loss, float* d_train_label, float* outa);

__global__ void kernelBackprop(float* outa, float* d_W2, float* d_W3, float* d_train_label, int n);

__global__ void kernelUpdate(float* d_W1, float* d_W2, float* d_W3, float* d_b1, float* d_b2, float* d_b3, float* d_train_data, int n, float* d_delta1, float* d_delta2, float* d_delta3, float* d_h1a, float* d_h2a);

// __global__ void kernelFull(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* d_train_data, float* d_train_label);