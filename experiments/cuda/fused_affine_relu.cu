// Small hardware experiment: one fused kernel vs two unfused kernels.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;} } while(0)
__global__ void affine(const float* x,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=2.0f*x[i]-0.5f;}
__global__ void relu(float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fmaxf(0.0f,y[i]);}
__global__ void fused(const float* x,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fmaxf(0.0f,2.0f*x[i]-0.5f);}
int main(){
 const int n=1<<20,iterations=200,threads=256,blocks=(n+threads-1)/threads;
 std::vector<float> host(n),a(n),b(n);for(int i=0;i<n;i++)host[i]=(i%1000)/500.f-1.f;
 float *x,*y,*z;CUDA(cudaMalloc(&x,n*sizeof(float)));CUDA(cudaMalloc(&y,n*sizeof(float)));CUDA(cudaMalloc(&z,n*sizeof(float)));
 CUDA(cudaMemcpy(x,host.data(),n*sizeof(float),cudaMemcpyHostToDevice));
 for(int i=0;i<20;i++){affine<<<blocks,threads>>>(x,y,n);relu<<<blocks,threads>>>(y,n);fused<<<blocks,threads>>>(x,z,n);}CUDA(cudaDeviceSynchronize());
 cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));float unfused_ms,fused_ms;
 CUDA(cudaEventRecord(start));for(int i=0;i<iterations;i++){affine<<<blocks,threads>>>(x,y,n);relu<<<blocks,threads>>>(y,n);}CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));CUDA(cudaEventElapsedTime(&unfused_ms,start,end));
 CUDA(cudaEventRecord(start));for(int i=0;i<iterations;i++)fused<<<blocks,threads>>>(x,z,n);CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));CUDA(cudaEventElapsedTime(&fused_ms,start,end));CUDA(cudaGetLastError());
 CUDA(cudaMemcpy(a.data(),y,n*sizeof(float),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(b.data(),z,n*sizeof(float),cudaMemcpyDeviceToHost));
 float error=0;for(int i=0;i<n;i++)error=std::max(error,std::fabs(a[i]-b[i]));
 printf("{\"elements\":%d,\"unfused_ms\":%.6f,\"fused_ms\":%.6f,\"max_absolute_error\":%.9f,\"scope\":\"kernel time, transfers excluded\"}\n",n,unfused_ms/iterations,fused_ms/iterations,error);
 CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));CUDA(cudaFree(x));CUDA(cudaFree(y));CUDA(cudaFree(z));return error<1e-6?0:1;
}
