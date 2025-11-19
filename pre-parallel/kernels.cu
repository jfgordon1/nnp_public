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
#include "config.h"


// __global__ void kernelForward(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* train_data){
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     float h1[256]; float h1a[256];
//     for (int j=row; j<row+1;j++){
//         h1[j]=d_b1[j];
//         for (int i=col; i<col+1;i++) h1[j]+=train_data[i]*d_W1[i*256+j];
//         h1a[j]=(h1[j] > 0 ? h1[j] : 0 );
//     }
//     float h2[128]; float h2a[128];
//     for (int j=row; j<row+1;j++){
//         h2[j]=d_b2[j];
//         for (int i=col; i<col+1;i++) h2[j]+=h1a[i]*d_W2[i*128+j];
//         h2a[j]=(h2[j] > 0 ? h2[j] : 0 );
//     }
//     float out[10]; float outa[10];
//     for (int k=row; k<row+1;k++){
//         out[k]=d_b3[k];
//         for (int j=col; j<col+1;j++) out[k]+=h2a[j]*d_W3[j*10+k];
//     }
//     softmax(out,outa,10);
// }


// __global__ void kernelLoss(float loss, float* train_label, float* outa){
//     int row = blockIdx.y * blockDim.y + threadIdx.y;

//     for (int k=0;k<10;k++)
//         loss -= train_label[k]*logf(outa[k]+1e-8f);
// }


// __global__ void kernelBackprop(float* outa, float* d_W2, float* d_W3, float* train_label){
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     float delta3[10];
//     for (int k=0;k<10;k++)
//         delta3[k] = train_label[k]-outa[k];

//     float delta2[128];
//     for (int j=0;j<128;j++){
//         float err=0;
//         for (int k=0;k<10;k++) err+=delta3[k]*d_W3[j*10+k];
//         delta2[j]=err*(h2a[j] > 0 ? 1 : 0);
//     }

//     float delta1[256];
//     for (int j=0;j<256;j++){
//         float err=0;
//         for (int k=0;k<128;k++) err+=delta2[k]*d_W2[j*128+k];
//         delta1[j]=err*(h1a[j] > 0 ? 1 : 0);
//     }
// }


// __global__ void kernelUpdate(float* d_W1, float* d_W2, float* d_W3, float* d_b1, float* d_b2, float* d_b3, float* train_data){
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     for (int j=0;j<128;j++)
//         for (int k=0;k<10;k++)
//             d_W3[j*10+k]+=0.01*delta3[k]*h2a[j];
//     for (int k=0;k<10;k++) d_b2[k]+=0.01*delta3[k];

//     for (int j=0;j<256;j++)
//         for (int k=0;k<128;k++)
//             d_W2[j*128+k]+=0.01*delta2[k]*h1a[j];
//     for (int k=0;k<128;k++) d_b2[k]+=0.01*delta2[k];

//     for (int i=0;i<784;i++)
//         for (int j=0;j<256;j++)
//             d_W1[i*256+j]+=0.01*delta1[j]*train_data[i];
//     for (int j=0;j<256;j++) d_b1[j]+=0.01*delta1[j];
// }

__global__ void kernelFull(float* d_W1, float* d_b1, float* d_W2, float* d_b2, float* d_W3, float* d_b3, float* train_data, float* train_label) {
    int row = blockIdx.y + blockDim.y + threadIdx.y;
    for (int epoch=row; epoch<row+1; epoch++) {
        if (epoch >=5) break;
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ---------- Forward ----------

            // kernelForward<<<blocksPerGrid, threadsPerBlock>>>(d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, train_data[n]);
            
            float h1[H1], h1a[H1];
            for (int j=0;j<H1;j++){
                h1[j]=model->b1[j];
                for (int i=0;i<SIZE;i++) h1[j]+=train_data[n][i]*model->W1[i*H1+j];
                h1a[j]=relu(h1[j]);
            }
            float h2[H2], h2a[H2];
            for (int j=0;j<H2;j++){
                h2[j]=model->b2[j];
                for (int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j];
                h2a[j]=relu(h2[j]);
            }
            float out[CLASSES], outa[CLASSES];
            for (int k=0;k<CLASSES;k++){
                out[k]=model->b3[k];
                for (int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k];
            }
            softmax(out,outa,CLASSES);

            // ---------- Loss ----------

            // kernelLoss<<<blocksPerGrid, threadsPerBlock>>>(loss, train_label[n], outa);

            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa[k]+1e-8f);


            // ---------- Backprop ----------

            // kernelBackprop<<<blocksPerGrid, threadsPerBlock>>>(outa, d_W2, d_W3, train_label[n]);

            float delta3[CLASSES];
            for (int k=0;k<CLASSES;k++)
                delta3[k] = train_label[n][k]-outa[k];

            float delta2[H2];
            for (int j=0;j<H2;j++){
                float err=0;
                for (int k=0;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
                delta2[j]=err*drelu(h2a[j]);
            }

            float delta1[H1];
            for (int j=0;j<H1;j++){
                float err=0;
                for (int k=0;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
                delta1[j]=err*drelu(h1a[j]);
            }

            // ---------- Update ----------

            // kernelUpdate<<<blocksPerGrid, threadsPerBlock>>>(d_W1, d_W2, d_W3, d_b1, d_b2, d_b3, train_data[n]);

            for (int j=0;j<H2;j++)
                for (int k=0;k<CLASSES;k++)
                    model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
            for (int k=0;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

            for (int j=0;j<H1;j++)
                for (int k=0;k<H2;k++)
                    model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
            for (int k=0;k<H2;k++) model->b2[k]+=LR*delta2[k];

            for (int i=0;i<SIZE;i++)
                for (int j=0;j<H1;j++)
                    model->W1[i*H1+j]+=LR*delta1[j]*train_data[n][i];
            for (int j=0;j<H1;j++) model->b1[j]+=LR*delta1[j];
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }
}