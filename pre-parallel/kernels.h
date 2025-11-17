/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();
__global__ void kernelForward(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* train_data)