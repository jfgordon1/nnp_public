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

    float h1[H1], h1a[H1];
    for (int j=row;j<H1;j++){
        h1[j]=d_b1[j];
        for (int i=col;i<col+1;i++) h1[j]+=train_data[i]*d_W1[i*H1+j];
        h1a[j]=relu(h1[j]);
    }
    float h2[H2], h2a[H2];
    for (int j=row;j<H2;j++){
        h2[j]=d_b2[j];
        for (int i=col;i<H1[i]+1;i++) h2[j]+=h1a[i]*d_W2[i*H2+j];
        h2a[j]=relu(h2[j]);
    }
    float out[CLASSES], outa[CLASSES];
    for (int k=row;k<CLASSES;k++){
        out[k]=d_b3[k];
        for (int j=col;j<H2[k]+1;j++) out[k]+=h2a[j]*d_W3[j*CLASSES+k];
    }
    softmax(out,outa,CLASSES);
}


__global__ void kernelLoss(float loss, float* train_label, float* outa){
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int k=row;k<CLASSES;k++)
        loss -= train_label[n][k]*logf(outa[k]+1e-8f);
}


__global__ void kernelBackprop(float* outa, float* d_W2, float* d_W3, float* train_data){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float delta3[CLASSES];
    for (int k=row;k<CLASSES;k++)
        delta3[k] = train_label[n][k]-outa[k];

    float delta2[H2];
    for (int j=row;j<H2;j++){
        float err=0;
        for (int k=col;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
        delta2[j]=err*drelu(h2a[j]);
    }

    float delta1[H1];
    for (int j=row;j<H1;j++){
        float err=0;
        for (int k=col;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
        delta1[j]=err*drelu(h1a[j]);
    }
}


__global__ void kernelUpdate(float* d_W1, float* d_W2, float* d_W3, float* train_data){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j=row;j<H2;j++)
        for (int k=col;k<CLASSES;k++)
            model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
    for (int k=col;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

    for (int j=row;j<H1;j++)
        for (int k=col;k<H2;k++)
            model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
    for (int k=col;k<H2;k++) model->b2[k]+=LR*delta2[k];

    for (int i=row;i<SIZE;i++)
        for (int j=col;j<H1;j++)
            model->W1[i*H1+j]+=LR*delta1[j]*train_data[n][i];
    for (int j=col;j<H1;j++) model->b1[j]+=LR*delta1[j];
}