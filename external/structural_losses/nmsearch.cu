__global__ void getNearest(int b,int n,float * xyz,int m,float *  xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*3+0]-x1;
							float y2=buf[k*3+1]-y1;
							float z2=buf[k*3+2]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*3+3]-x1;
							float y2=buf[k*3+4]-y1;
							float z2=buf[k*3+5]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*3+6]-x1;
							float y2=buf[k*3+7]-y1;
							float z2=buf[k*3+8]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*3+9]-x1;
							float y2=buf[k*3+10]-y1;
							float z2=buf[k*3+11]-z1;
							float d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*3+0]-x1;
					float y2=buf[k*3+1]-y1;
					float z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
#include <cstdio>
#include <time.h>
static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}
void getNearestLauncher(int b,int n,float * xyz,int m,float * xyz2,float * result,int * result_i){
	float * xyz_g;
	float * xyz2_g;
	float * result_g;
	int * result_i_g;
	double t0=get_time();
	cudaMalloc(&xyz_g,b*n*3*4);
	cudaMalloc(&xyz2_g,b*m*3*4);
	cudaMalloc(&result_g,b*n*4);
	cudaMalloc(&result_i_g,b*n*4);
	cudaMemcpy(xyz_g,xyz,b*n*3*4,cudaMemcpyHostToDevice);
	cudaMemcpy(xyz2_g,xyz2,b*m*3*4,cudaMemcpyHostToDevice);
	double t1=get_time();
	getNearest<<<dim3(32,16,1),512>>>(b,n,xyz_g,m,xyz2_g,result_g,result_i_g);
	cudaDeviceSynchronize();
	double t2=get_time();
	cudaMemcpy(result,result_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(result_i,result_i_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaFree(xyz_g);
	cudaFree(xyz2_g);
	cudaFree(result_g);
	cudaFree(result_i_g);
	double t3=get_time();
	printf("time %f %f %f %f\n",t3-t0,t1-t0,t2-t1,t3-t2);
}
void getNearestLauncher2(int b,int n,float * xyz,int m,float * xyz2,float * result,int * result_i,float * result2,int * result2_i){
	float * xyz_g;
	float * xyz2_g;
	float * result_g;
	int * result_i_g;
	float * result2_g;
	int * result2_i_g;
	double t0=get_time();
	cudaMalloc(&xyz_g,b*n*3*4);
	cudaMalloc(&xyz2_g,b*m*3*4);
	cudaMalloc(&result_g,b*n*4);
	cudaMalloc(&result_i_g,b*n*4);
	cudaMalloc(&result2_g,b*n*4);
	cudaMalloc(&result2_i_g,b*n*4);
	cudaMemcpy(xyz_g,xyz,b*n*3*4,cudaMemcpyHostToDevice);
	cudaMemcpy(xyz2_g,xyz2,b*m*3*4,cudaMemcpyHostToDevice);
	double t1=get_time();
	getNearest<<<dim3(32,16,1),512>>>(b,n,xyz_g,m,xyz2_g,result_g,result_i_g);
	getNearest<<<dim3(32,16,1),512>>>(b,m,xyz2_g,n,xyz_g,result2_g,result2_i_g);
	cudaDeviceSynchronize();
	double t2=get_time();
	cudaMemcpy(result,result_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(result_i,result_i_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(result2,result2_g,b*m*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(result2_i,result2_i_g,b*m*4,cudaMemcpyDeviceToHost);
	cudaFree(xyz_g);
	cudaFree(xyz2_g);
	cudaFree(result_g);
	cudaFree(result_i_g);
	cudaFree(result2_g);
	cudaFree(result2_i_g);
	double t3=get_time();
	printf("time %f %f %f %f\n",t3-t0,t1-t0,t2-t1,t3-t2);
}
