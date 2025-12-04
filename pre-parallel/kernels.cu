/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include "config.h"
#include "kernels.h"

__host__ __device__ void softmax(float* z, float* out, int len){
    float max = z[0];
    for (int i = 1; i<len;i++) if (z[i]>max) max = z[i];
    float sum = 0;
    for (int i=0;i<len;i++) { out[i] = expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

__host__ __device__ float relu(float x) { return x > 0 ? x : 0; }

__host__ __device__ float drelu(float y) { return y > 0 ? 1 : 0; }

__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1, float* d_h1a, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    d_h1[j]=d_b1[j];
    for (int i=0; i<SIZE; i++) d_h1[j]+=d_train_data[n*SIZE+i]*d_W1[i*H1+j];
    d_h1a[j]=relu(d_h1[j]);
}

__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2, float* d_h2a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    d_h2[j]=d_b2[j];
    for (int i=0; i<H1; i++) d_h2[j]+=d_h1a[i]*d_W2[i*H2+j];
    d_h2a[j]=relu(d_h2[j]);
}

__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out, float* d_outa) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= CLASSES) return;
    d_out[j]=d_b3[j];
    for (int i=0; i<H2; i++) d_out[i]+=d_h2a[i]*d_W3[i*CLASSES+j];
    __syncthreads();
    if (j == 0) softmax(d_out, d_outa, CLASSES);
}

__global__ void sumSubLoss(float* d_train_label, float* d_outa, float* d_loss, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= CLASSES) return;
    *d_loss -= d_train_label[n*CLASSES+j]*logf(d_outa[j]+1e-8f);
}

__global__ void vectorAssignDelta3(float* d_delta3, float* d_train_label, float* d_outa, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= CLASSES) return;
    d_delta3[j] = d_train_label[n*SIZE+j]-d_outa[j];
}

__global__ void vectorMatrixMultDelta2(float* d_delta2, float* d_delta3, float* d_W3, float* d_h2a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    float err=0;
    for (int i=0; i<CLASSES; i++) err+=d_delta3[i]*d_W3[j*CLASSES+i];
    __syncthreads();
    d_delta3[j]=err*drelu(d_h2a[j]);
}

__global__ void vectorMatrixMultDelta1(float* d_delta1, float* d_delta2, float* d_W2, float* d_h1a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    float err=0;
    for (int i=0; i<H2; i++) err+=d_delta2[i]*d_W2[j*H2+i];
    __syncthreads();
    d_delta1[j]=err*drelu(d_h1a[j]);
}

__global__ void vectorMatrixMultB3(float* d_W3, float* d_delta3, float* d_h2a, float* d_b3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    for (int i=0; i<CLASSES; i++) {
        d_W3[j*CLASSES+i]+=LR*d_delta3[i]*d_h2a[j];
        d_b3[i]+=LR*d_delta3[i];
    }
}

__global__ void vectorMatrixMultB2(float* d_W2, float* d_delta2, float* d_h1a, float* d_b2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    for (int i=0; i<H2; i++) {
        d_W2[j*H2+i]+=LR*d_delta2[i]*d_h1a[j];
        d_b2[i]+=LR*d_delta2[i];
    }
}

__global__ void vectorMatrixMultB1(float* d_W1, float* d_delta1, float* d_train_data, float* d_b1, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= SIZE) return;
    for (int i=0; i<H1; i++) {
        d_W1[j*H1+i]+=LR*d_delta1[i]*d_train_data[n*SIZE+j];
        d_b1[i]+=LR*d_delta1[i];
    }
}