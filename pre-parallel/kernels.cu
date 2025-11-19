/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include "kernels.h"


__global__ void kernelForward(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* train_data){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float h1[H1]; float h1a[H1];
    for (int j=row; j<row+1;j++){
        h1[j]=d_b1[j];
        for (int i=col; i<col+1;i++) h1[j]+=train_data[i]*d_W1[i*H1+j];
        h1a[j]=relu(h1[j]);
    }
    float h2[H2]; float h2a[H2];
    for (int j=row; j<j+1;j++){
        h2[j]=d_b2[j];
        for (int i=col; i<H1[i]+1;i++) h2[j]+=h1a[i]*d_W2[i*H2+j];
        h2a[j]=relu(h2[j]);
    }
    float out[CLASSES]; float outa[CLASSES];
    for (int k=row; k<k+1;k++){
        out[k]=d_b3[k];
        for (int j=col; j<H2[k]+1;j++) out[k]+=h2a[j]*d_W3[j*CLASSES+k];
    }
    softmax(out,outa,CLASSES);
}



