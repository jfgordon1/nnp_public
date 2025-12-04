/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */


#include <cuda.h>
#include <math.h>
#include "config.h"
#include "kernels.h"

__global__ void kernelSoftMax(float* z, float* out, int len){
  float max = z[0];
  for (int i = 1; i<len;i++) if (z[i]>max) max = z[i];
  float sum = 0;
  for (int i=0;i<len;i++) { out[i] = expf(z[i]-max); sum+=out[i]; }
  for (int i=0;i<len;i++) out[i]/=sum;
}

__device__ inline float kernelRelu(float x) { return x > 0 ? x : 0; }

__device__ inline float kernelDRelu(float y) { return y > 0 ? 1 : 0; }

__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1, float* d_h1a){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    d_h1[j]=d_b1[j];
    for (int i=0;i<SIZE;i++) d_h1[j]+=d_train_data[i]*d_W1[i*H1+j];
    d_h1a[j]=kernelRelu(d_h1[j]);
}

__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2, float* d_h2a){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    d_h2[j]=d_b2[j];
    for (int i=0;i<H1;i++) d_h2[j]+=d_h1a[i]*d_W2[i*H2+j];
    d_h2a[j]=kernelRelu(d_h2[j]);
}

__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= CLASSES) return;
    d_out[j]=d_b3[j];
    for (int i=0;i<H2;i++) d_out[j]+=d_h2a[i]*d_W3[i*CLASSES+j];
}

__global__ void kernelLoss(float* d_outa, float* d_train_label, float* d_loss){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(d_loss, -d_train_label[k]*logf(d_outa[k]+1e-8f));
}

__global__ void delta3Backprop(float* d_outa, float* d_train_label, float* d_delta3){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    d_delta3[k] = d_train_label[k]-d_outa[k];
}

__global__ void delta2Backprop(float* d_delta3, float* d_W3, float* d_h2a, float* d_delta2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err=0;
    for (int k=0;k<CLASSES;k++) err+=d_delta3[k]*d_W3[j*CLASSES+k];
    d_delta2[j]=err*kernelDRelu(d_h2a[j]);
}

__global__ void delta1Backprop(float* d_delta2, float* d_W2, float* d_h1a, float* d_delta1){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err=0;
    for (int k=0;k<H2;k++) err+=d_delta2[k]*d_W2[j*H2+k];
    d_delta1[j]=err*kernelDRelu(d_h1a[j]);
}

__global__ void updateClasses(float* d_delta3, float* d_h2a, float* d_W3){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k=0;k<CLASSES;k++){
        d_W3[j*CLASSES+k]+=LR*d_delta3[k]*d_h2a[j];
    }
}

__global__ void updateH2(float* d_delta2, float* d_h1a, float* d_W2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k=0;k<H2;k++){
        d_W2[j*H2+k]+=LR*d_delta2[k]*d_h1a[j];
    }
}

__global__ void updateH1(float* d_delta1, float* d_train_data, float* d_W1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j=0;j<H1;j++){
        d_W1[i*H1+j]+=LR*d_delta1[j]*d_train_data[i];
    }
}

__global__ void updateBias3(float* d_delta3, float* d_b3){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    d_b3[k]+=LR*d_delta3[k];
}

__global__ void updateBias2(float* d_delta2, float* d_b2){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    d_b2[k]+=LR*d_delta2[k];
}

__global__ void updateBias1(float* d_delta1, float* d_b1){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    d_b1[j]+=LR*d_delta1[j];
}