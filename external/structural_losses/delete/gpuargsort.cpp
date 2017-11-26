#include <cstdio>
#include <algorithm>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
using namespace std;
void batchSortLauncher(int b,int n,float * inp,float * out,int * out_i);
float randomf(){
	return (rand()+0.5)/(RAND_MAX+1.0);
}
static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}
int main()
{
	cudaSetDevice(2);
	const int b=32,n=4096;
	float * inp=new float[b*n];
	float * inp2=new float[b*n];
	float * out=new float[b*n];
	int * out_i=new int[b*n];
	for (int i=0;i<b*n;i++)
		inp[i]=randomf();
	memcpy(inp2,inp,b*n*4);
	double besttime=0;
	for (int run=0;run<30;run++){
		double t0=get_time();
		batchSortLauncher(b,n,inp,out,out_i);
		double t1=get_time();
		printf("time=%f\n",t1-t0);
		if (run==0 || t1-t0<besttime){
			besttime=t1-t0;
		}
	}
	int errorcnt=0;
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++){
			int x=out_i[i*n+j];
			if (x<0 || x>=n || inp2[i*n+x]!=out[i*n+j]){
				printf("invalid index i=%d j=%d x=%d out=%f\n",i,j,x,out[i*n+j]);
				errorcnt++;
			}else{
				inp2[i*n+x]=randomf();
				if (j && out[i*n+j]<out[i*n+j-1]){
					printf("order violation i=%d j=%d x=%d out=%f out_prev=%f\n",i,j,x,out[i*n+j],out[i*n+j-1]);
					errorcnt++;
				}
			}
			if (errorcnt>=20)
				break;
		}
		if (errorcnt>=20)
			break;
	}
	printf("besttime=%f\n",besttime);
	return 0;
}

