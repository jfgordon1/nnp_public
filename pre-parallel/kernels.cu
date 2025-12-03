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

__global__ inline float relu(float x) { return x > 0 ? x : 0; }

__global__ inline float drelu(float y) { return y > 0 ? 1 : 0; }
