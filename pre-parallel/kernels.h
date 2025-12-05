/* 
 * kernels.h
 *  Jacob Gordon
 *  Tristan Dendorfer
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel()
__host__ __device__ float relu(float x);
__host__ __device__ float drelu(float y);
__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1, float* d_h1a, int n);
__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2, float* d_h2a);
__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out, float* d_outa);
__global__ void sumSubLoss(float* d_train_label, float* d_outa, float* loss, int n);
__global__ void vectorAssignDelta3(float* d_delta3, float* d_train_label, float* d_outa, int n);
__global__ void vectorMatrixMultDelta2(float* d_delta2, float* d_delta3, float* d_W3, float* d_h2a);
__global__ void vectorMatrixMultDelta1(float* d_delta1, float* d_delta2, float* d_W2, float* d_h1a);
__global__ void vectorMatrixMultB3(float* d_W3, float* d_delta3, float* d_h2a, float* d_b3);
__global__ void vectorMatrixMultB2(float* d_W2, float* d_delta2, float* d_h1a, float* d_b2);
__global__ void vectorMatrixMultB1(float* d_W1, float* d_delta1, float* d_train_data, float* d_b1, int n);
