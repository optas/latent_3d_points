#include <cstdio>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <math.h>
using namespace std;
void cumsumLauncher(int b,int n,const float * inp,float * out);
float randomf(){
	return (rand()+0.5)/(RAND_MAX+1.0);
}
static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}
int main_cumsum()
{
	int b=32,n=16131;
	float * inp=new float[b*n];
	float * out=new float[b*n];
	float * std=new float[b*n];
	for (int i=0;i<b*n;i++){
		inp[i]=randomf();
	}
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++){
			s+=inp[i*n+j];
			std[i*n+j]=s;
		}
	}
	float * inp_g, *out_g;
	cudaSetDevice(2);
	cudaMalloc(&inp_g,b*n*4);
	cudaMalloc(&out_g,b*n*4);
	cudaMemcpy(inp_g,inp,b*n*4,cudaMemcpyHostToDevice);
	double besttime=0;
	for (int run=0;run<20;run++){
		double t2=get_time();
		for (int i=0;i<1000;i++){
			cumsumLauncher(b,n,inp_g,out_g);
		}
		cudaDeviceSynchronize();
		double t=(get_time()-t2)/1000.0;
		printf("run %d time %f\n",run,t);
		if (run==0 || t<besttime)
			besttime=t;
	}
	printf("besttime=%f\n",besttime);
	cudaMemcpy(out,out_g,b*n*4,cudaMemcpyDeviceToHost);
	cudaFree(inp_g);
	cudaFree(out_g);
	double maxerr=0;
	for (int i=0;i<b*n;i++)
		maxerr=max(maxerr,fabs(out[i]-std[i]));
	printf("maxerror=%f\n",maxerr);
	printf("%f %f %f %f\n",out[0],out[4096],std[0],std[4096]);
	printf("%f %f %f %f\n",out[0],out[8192],std[0],std[8192]);
	delete []inp;
	delete []out;
	delete []std;
	return 0;
}
void probSampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out);
int main_probsample(){
	int b=32,n=128,m=16313;
	float * inp_p=new float[b*n];
	float * inp_r=new float[b*m];
	int * out=new int[b*m];
	int * std=new int[b*m];
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++){
			inp_p[i*n+j]=(i+j)%n+0.5;
		}
	}
	for (int i=0;i<b*m;i++){
		inp_r[i]=randomf();
	}
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			s+=inp_p[i*n+j];
		for (int j=0;j<m;j++){
			int c=0;
			double target=inp_r[i*m+j]*s;
			for (int k=0;k<n;k++){
				if (target>=inp_p[i*n+k]){
					c++;
					target-=inp_p[i*n+k];
				}else{
					break;
				}
			}
			std[i*m+j]=c;
		}
	}
	float * inp_p_g;
	float * inp_r_g;
	float * temp_g;
	int * out_g;
	cudaMalloc(&inp_p_g,b*n*4);
	cudaMalloc(&inp_r_g,b*m*4);
	cudaMalloc(&temp_g,b*n*4);
	cudaMalloc(&out_g,b*m*4);
	cudaMemset(out_g,0,b*m*4);
	cudaMemcpy(inp_p_g,inp_p,b*n*4,cudaMemcpyHostToDevice);
	cudaMemcpy(inp_r_g,inp_r,b*m*4,cudaMemcpyHostToDevice);
	double besttime=0;
	for (int run=0;run<10;run++){
		double t2=get_time();
		for (int i=0;i<100;i++){
			probSampleLauncher(b,n,m,inp_p_g,inp_r_g,temp_g,out_g);
		}
		cudaDeviceSynchronize();
		double t=(get_time()-t2)/100.0;
		printf("run %d time %f\n",run,t);
		if (run==0 || t<besttime)
			besttime=t;
	}
	//float * temp=new float[b*n];
	//cudaMemcpy(temp,temp_g,b*n*4,cudaMemcpyDeviceToHost);
	//for (int i=0;i<n;i++)
		//printf("temp %d %f\n",i,temp[i]);

	cudaMemcpy(out,out_g,b*m*4,cudaMemcpyDeviceToHost);
	/*int * cnt=new int[b*n];
	for (int i=0;i<b*n;i++)
		cnt[i]=0;
	for (int i=0;i<b;i++){
		for (int j=0;j<m;j++){
			if (out[i*m+j]<0 || out[i*m+j]>=n){
				printf("Invalid index i=%d j=%d %d\n",i,j,out[i*m+j]);
			}else{
				cnt[i*n+out[i*m+j]]++;
			}
		}
	}
	for (int i=0;i<b;i++){
		int k=rand()%n;
		//for (int k=0;k<n;k++){
			printf("i=%d k=%d %f %f\n",i,k,((i+k)%n+0.5)/double(n)/double(n/2.0),double(cnt[i*n+k])/m);
		//}
	}*/
	int good=0,bad=0;
	for (int i=0;i<b*m;i++){
		if (std[i]==out[i])
			good++;
		else{
			bad++;
			//break;
		}
	}
	printf("good=%d bad=%d\n",good,bad);
	printf("besttime=%f\n",besttime);
	return 0;
}
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out);
int main_farthestpoint(){
	int b=32,n=16384,m=1024;
	float * inp=new float[b*n*3];
	int * out=new int[b*m];
	int * std=new int[b*m];
	for (int i=0;i<b*n*3;i++)
		inp[i]=randomf();
	float * temp=new float[n];
	for (int i=0;i<b;i++){
		int old=0;
		std[i*m+0]=old;
		for (int j=0;j<n;j++){
			temp[j]=(
				(inp[(i*n+j)*3+0]-inp[(i*n+old)*3+0])*(inp[(i*n+j)*3+0]-inp[(i*n+old)*3+0])+
				(inp[(i*n+j)*3+1]-inp[(i*n+old)*3+1])*(inp[(i*n+j)*3+1]-inp[(i*n+old)*3+1])+
				(inp[(i*n+j)*3+2]-inp[(i*n+old)*3+2])*(inp[(i*n+j)*3+2]-inp[(i*n+old)*3+2])
			);
		}
		for (int j=1;j<m;j++){
			int a=0;
			float d2=temp[0];
			for (int k=1;k<n;k++){
				if (temp[k]>d2){
					d2=temp[k];
					a=k;
				}
			}
			std[i*m+j]=a;
			for (int k=0;k<n;k++){
				temp[k]=min(temp[k],(
					(inp[(i*n+k)*3+0]-inp[(i*n+a)*3+0])*(inp[(i*n+k)*3+0]-inp[(i*n+a)*3+0])+
					(inp[(i*n+k)*3+1]-inp[(i*n+a)*3+1])*(inp[(i*n+k)*3+1]-inp[(i*n+a)*3+1])+
					(inp[(i*n+k)*3+2]-inp[(i*n+a)*3+2])*(inp[(i*n+k)*3+2]-inp[(i*n+a)*3+2])
				));
			}
		}
	}
	float * inp_g;
	float * temp_g;
	int * out_g;
	cudaMalloc(&inp_g,b*n*3*4);
	cudaMalloc(&temp_g,32*n*4);
	cudaMalloc(&out_g,b*m*4);
	cudaMemset(out_g,0,b*m*4);
	cudaMemcpy(inp_g,inp,b*n*3*4,cudaMemcpyHostToDevice);
	double besttime=0;
	for (int run=0;run<20;run++){
		double t2=get_time();
		for (int i=0;i<10;i++){
			farthestpointsamplingLauncher(b,n,m,inp_g,temp_g,out_g);
		}
		cudaDeviceSynchronize();
		double t=(get_time()-t2)/10.0;
		printf("run %d time %f\n",run,t);
		if (run==0 || t<besttime)
			besttime=t;
	}
	cudaMemcpy(out,out_g,b*m*4,cudaMemcpyDeviceToHost);
	cudaFree(inp_g);
	cudaFree(temp_g);
	cudaFree(out_g);
	int good=0,bad=0;
	for (int i=0;i<b*m;i++){
		if (std[i]==out[i])
			good++;
		else{
			//printf("i=%d %d %d %d\n",i/m,i%m,std[i],out[i]);
			bad++;
			//break;
		}
	}
	printf("good=%d bad=%d\n",good,bad);
	printf("besttime=%f\n",besttime);
	return 0;
}
int main(){
	//return main_cumsum();
	//return main_probsample();
	return main_farthestpoint();
}

