/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();
__global__ void softmax(flaot* z, float* out, int len);
__global__ inline float relu(float x);
__global__ inline float drelu(float y);
__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1);
__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2);
__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out);
__global__ void kernelLoss(float* d_outa, float* d_train_label, float* d_loss);
__global__ void delta3Backprop(float* d_outa, float* d_train_label, float* d_delta3);
__global__ void delta2Backprop(float* d_delta3, float* d_W3, float* d_h2a, float* d_delta2);
__global__ void delta1Backprop(float* d_delta2, float* d_W2, float* d_h1a, float* d_delta1);
__global__ void updateClasses(float* d_delta3, float* d_h2a, float* d_W3);
__global__ void updateH2(float* d_delta2, float* d_h1a, float* d_W2);
__global__ void updateH1(float* d_delta1, float* d_train_data, float* d_W1);
__global__ void updateBias3(float* d_delta3, float* d_b3);
__global__ void updateBias2(float* d_delta2, float* d_b2);
__global__ void updateBias1(float* d_delta1, float* d_b1);