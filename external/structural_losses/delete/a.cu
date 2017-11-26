#include <cstdio>
#include <time.h>
#include <algorithm>
#include <string.h>
#include <sys/time.h>
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
double gettime(){
	timeval tv;
	gettimeofday(&tv,NULL);
	return double(tv.tv_sec)+tv.tv_usec*0.000001;
}
int main()
{
	const int N=160,M=16384,Q=1024;
	FILE * fin=fopen("binpoints","rb");
	float * dataset=new float[N*M*4];
	float * queries=new float[N*Q*4];
	fread(dataset,3*4,N*M,fin);
	for (int i=N*M-1;i>=0;i--){
		dataset[i*4+3]=0;
		dataset[i*4+2]=dataset[i*3+2];
		dataset[i*4+1]=dataset[i*3+1];
		dataset[i*4+0]=dataset[i*3+0];
	}
	fread(queries,3*4,N*Q,fin);
	for (int i=N*Q-1;i>=0;i--){
		queries[i*4+3]=0;
		queries[i*4+2]=queries[i*3+2];
		queries[i*4+1]=queries[i*3+1];
		queries[i*4+0]=queries[i*3+0];
	}
	fclose(fin);
	int * result1=new int[N*Q];
	float * distances1=new float[N*Q];
	int * result2=new int[N*M];
	float * distances2=new float[N*M];
	float *dataset_g;
	float *queries_g;
	cudaMalloc(&dataset_g,N*M*4*4);
	cudaMalloc(&queries_g,N*Q*4*4);
	//int *id_g;
	//float *dist_g;
	//cudaMalloc(&id_g,N*std::max(M,Q)*4);
	//cudaMalloc(&dist_g,N*std::max(M,Q)*4);
	cudaMemcpy(dataset_g,dataset,N*M*4*4,cudaMemcpyHostToDevice);
	cudaMemcpy(queries_g,queries,N*Q*4*4,cudaMemcpyHostToDevice);
	//cudaMemcpy(id_g,result,N*M*4,cudaMemcpyHostToDevice);
	//cudaMemcpy(dist_g,distances,N*Q*4,cudaMemcpyHostToDevice);
	double t0=gettime();
	for (int i=0;i<N;i++){
		printf("i=%d\n",i);
		if (true){
			flann::Matrix<float> m(dataset_g+i*M*4,M,3,4*4);
			flann::Matrix<float> q(queries_g+i*Q*4,Q,3,4*4);
			flann::Matrix<int> id(result1+i*Q,Q,1);
			flann::Matrix<float> dist(distances1+i*Q,Q,1);
			//flann::Index<flann::L2<float> > flannindex(m,flann::KDTreeIndexParams(4));
			flann::KDTreeCuda3dIndexParams params;
			params["input_is_gpu_float4"]=true;
			flann::Index<flann::L2<float> > flannindex(m,params);
			flannindex.buildIndex();
			flann::SearchParams sparams;
			sparams.matrices_in_gpu_ram=false;
			flannindex.knnSearch(q,id,dist,1,sparams);
		}
		if (true){
			flann::Matrix<float> m(dataset_g+i*Q*4,Q,3,4*4);
			flann::Matrix<float> q(queries_g+i*M*4,M,3,4*4);
			flann::Matrix<int> id(result2+i*M,M,1);
			flann::Matrix<float> dist(distances2+i*M,M,1);
			flann::KDTreeCuda3dIndexParams params;
			params["input_is_gpu_float4"]=true;
			//flann::Index<flann::L2<float> > flannindex(m,flann::KDTreeIndexParams(1));
			flann::Index<flann::L2<float> > flannindex(m,params);
			flannindex.buildIndex();
			flann::SearchParams sparams;
			sparams.matrices_in_gpu_ram=false;
			flannindex.knnSearch(q,id,dist,1,sparams);
		}
		printf("i %d time %f\n",i,(gettime()-t0)/(i+1));
	}
	cudaFree(dataset_g);
	cudaFree(queries_g);
	double mx=0;
	for (int i=0;i<N*M;i++)
		mx=max(mx,distances2[i]);
	printf("mx=%f\n",mx);
	mx=0;
	for (int i=0;i<N*Q;i++)
		mx=max(mx,distances1[i]);
	printf("mx=%f\n",mx);
	//cudaFree(id_g);
	//cudaFree(dist_g);
	return 0;
}

