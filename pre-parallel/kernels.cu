/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */


__global__ void softmax(flaot* z, float* out, int len){
  float max = z[0];
  for (int i = 1; i<len;i++) if (z[i]>max) max = z[i];
  float sum = 0;
  for (int i=0;i<len;i++) { out[i] = expf(z[i]-max); sum+=out[i]; }
  for (int i=0;i<len;i++) out[i]/=sum;
}

__host__ inline float relu(float x) { return x > 0 ? x : 0; }

__host__ inline float drelu(float y) { return y > 0 ? 1 : 0; }

__global__ void vectorMatrixMultH1(float* d_train_data, float* d_W1, float* d_b1, float* d_h1){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    d_h1[j]=d_b1[j];
    for (int i=0;i<SIZE;i++) d_h1[j]+=d_train_data[i]*d_W1[i*H1+j];
    d_h1a[j]=relu(d_h1[j]);
}

__global__ void vectorMatrixMultH2(float* d_h1a, float* d_W2, float* d_b2, float* d_h2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    d_h2[j]=d_b2[j];
    for (int i=0;i<H1;i++) d_h2[j]+=d_h1a[i]*d_W2[i*H2+j];
    d_h2a[j]=relu(d_h2[j]);
}

__global__ void vectorMatrixMultOut(float* d_h2a, float* d_W3, float* d_b3, float* d_out){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= CLASSES) return;
    d_out[j]=d_b3[j];
    for (int i=0;i<H2;i++) d_out[j]+=d_h2a[i]*d_W3[i*CLASSES+j];
}

__global__ void kernelLoss(float* d_outa, float* d_train_label, float* d_loss){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    d_loss -= d_train_label[k]*logf(d_outa[k]+1e-8f);
}

__global__ void delta3Backprop(d_outa, d_train_label, d_delta3){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    d_delta3[k] = d_train_label[k]-d_outa[k];
}

__global__ void delta2Backprop(d_delta3, d_W3, d_h2a, d_delta2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err=0;
    for (int k=0;k<CLASSES;k++) err+=d_delta3[k]*d_W3[j*CLASSES+k];
    d_delta2[j]=err*drelu(d_h2a[j]);
}

__global__ void delta1Backprop(d_delta2, d_W2, d_h1a, d_delta1){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err=0;
    for (int k=0;k<H2;k++) err+=d_delta2[k]*d_W2[j*H2+k];
    d_delta1[j]=err*drelu(d_h1a[j]);
}
