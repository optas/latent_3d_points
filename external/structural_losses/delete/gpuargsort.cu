#include <thrust/sort.h>
#include <thrust/device_vector.h>

void batchSortLauncher(int b,int n,float * inp,float * out,int * out_i){
	float * out_g;
	int * out_i_g;
	cudaMalloc(&out_g,b*n*4);
	cudaMalloc(&out_i_g,b*n*4);
	cudaMemcpy(out_g,inp,b*n*4,cudaMemcpyHostToDevice);
	for (int i=0;i<b;i++){
		thrust::sequence(thrust::device_ptr<int>(out_i_g+i*n),thrust::device_ptr<int>(out_i_g+(i*n+n)));
	}
	for (int i=0;i<b;i++){
		thrust::sort_by_key(thrust::device_ptr<float>(out_g+i*n),thrust::device_ptr<float>(out_g+(i*n+n)),thrust::device_ptr<int>(out_i_g+i*n));
	}
	cudaMemcpy(out,out_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(out_i,out_i_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaFree(out_g);
	cudaFree(out_i_g);
}
