/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();
__global__ void softmax(flaot* z, float* out, int len)
__global__ inline float relu(float x)
__global__ inline float drelu(float y)
