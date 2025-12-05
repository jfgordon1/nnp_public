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

__host__ __device__ float relu(float x) { return x > 0 ? x : 0; }

__host__ __device__ float drelu(float y) { return y > 0 ? 1 : 0; }

__global__ void findMax(float* d_vec, float* d_max) {
    int j = threadIdx.x;
    if (j >= CLASSES) return;
    float max = d_vec[0];
    __syncthreads();
    if (d_vec[j]>max) max=d_vec[j];
    __syncthreads();
    d_max = &max;
}

__global__ void kernelSoftMax(float* d_out, float* d_outa, float* d_max) {
    int j = threadIdx.x;
    if (j >= CLASSES) return;
    float sum=0;
    d_outa[j]=expf(d_out[j]-d_max);
    sum += d_outa[j];
    __syncthreads();
    d_outa[j]/=sum;
}

__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1, float* d_h1a, int n) {
    int j = threadIdx.x;
    if (j >= H1) return;
    float temp = d_b1[j];
    //d_h1[j]=d_b1[j];
    for (int i=0; i<SIZE; i++) temp+=d_train_data[n*SIZE+i]*d_W1[i*H1+j];
    d_h1[j]=temp;
    d_h1a[j]=relu(d_h1[j]);
}

__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2, float* d_h2a) {
    int j = threadIdx.x;
    if (j >= H2) return;
    float temp = d_b2[j];
    //d_h2[j]=d_b2[j];
    for (int i=0; i<H1; i++) temp+=d_h1a[i]*d_W2[i*H2+j];
    d_h2[j]=temp;
    d_h2a[j]=relu(d_h2[j]);
}

__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out, float* d_outa) {
    int j = threadIdx.x;
    if (j >= CLASSES) return;
    float temp = d_b3[j];
    //d_out[j]=d_b3[j];
    for (int i=0; i<H2; i++) temp+=d_h2a[i]*d_W3[i*CLASSES+j];
    d_out[j]=temp;
    __syncthreads();
}

__global__ void sumSubLoss(float* d_train_label, float* d_outa, float* d_loss, int n) {
    int j = threadIdx.x;
    if (j >= CLASSES) return;
    atomicAdd(d_loss, -d_train_label[n*CLASSES+j]*logf(d_outa[j]+1e-8f));
}

__global__ void vectorAssignDelta3(float* d_delta3, float* d_train_label, float* d_outa, int n) {
    int j = threadIdx.x;
    if (j >= CLASSES) return;
    d_delta3[j] = d_train_label[n*CLASSES+j]-d_outa[j];
}

__global__ void vectorMatrixMultDelta2(float* d_delta2, float* d_delta3, float* d_W3, float* d_h2a) {
    int j = threadIdx.x;
    if (j >= H2) return;
    float err=0;
    for (int i=0; i<CLASSES; i++) err+=d_delta3[i]*d_W3[j*CLASSES+i];
    d_delta2[j]=err*drelu(d_h2a[j]);
}

__global__ void vectorMatrixMultDelta1(float* d_delta1, float* d_delta2, float* d_W2, float* d_h1a) {
    int j = threadIdx.x;
    if (j >= H1) return;
    float err=0;
    for (int i=0; i<H2; i++) err+=d_delta2[i]*d_W2[j*H2+i];
    d_delta1[j]=err*drelu(d_h1a[j]);
}

__global__ void vectorMatrixMultB3(float* d_W3, float* d_delta3, float* d_h2a, float* d_b3) {
    int j = threadIdx.x;
    if (j >= H2) return;
    for (int i=0; i<CLASSES; i++) d_W3[j*CLASSES+i]+=LR*d_delta3[i]*d_h2a[j];
    if (j < CLASSES) d_b3[j]+=LR*d_delta3[j];
}

__global__ void vectorMatrixMultB2(float* d_W2, float* d_delta2, float* d_h1a, float* d_b2) {
    int j = threadIdx.x;
    if (j >= H1) return;
    for (int i=0; i<H2; i++) d_W2[j*H2+i]+=LR*d_delta2[i]*d_h1a[j];
    if (j < H2) d_b2[j]+=LR*d_delta2[j];
}

__global__ void vectorMatrixMultB1(float* d_W1, float* d_delta1, float* d_train_data, float* d_b1, int n) {
    int j = threadIdx.x;
    if (j >= SIZE) return;
    for (int i=0; i<H1; i++) d_W1[j*H1+i]+=LR*d_delta1[i]*d_train_data[n*SIZE+j];
    if (j < H1) d_b1[j]+=LR*d_delta1[j];
}