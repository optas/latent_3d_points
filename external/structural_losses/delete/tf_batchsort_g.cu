#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

__global__ void BatchSortGradKernel(int b,int n,const float * __restrict__ grad_out,const int * __restrict__ out_i,float * __restrict__ grad_inp){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<n;j+=gridDim.y*blockDim.x){
			int idx=out_i[i*n+j];
			grad_inp[i*n+idx]=grad_out[i*n+j];
		}
	}
}
void BatchSortGradKernelLauncher(int b,int n,const float * grad_out,const int * out_i,float * grad_inp){
	BatchSortGradKernel<<<dim3(1,16,1),256>>>(b,n,grad_out,out_i,grad_inp);
}
void BatchSortKernelLauncher(int b,int n,const float * inp,float * out,int * out_i){
	cudaMemcpy(out,inp,b*n*4,cudaMemcpyDeviceToDevice);
	for (int i=0;i<b;i++){
		thrust::sequence(thrust::device_ptr<int>(out_i+i*n),thrust::device_ptr<int>(out_i+(i*n+n)));
	}
	for (int i=0;i<b;i++){
		thrust::sort_by_key(thrust::device_ptr<float>(out+i*n),thrust::device_ptr<float>(out+(i*n+n)),thrust::device_ptr<int>(out_i+i*n));
	}
}

#endif
