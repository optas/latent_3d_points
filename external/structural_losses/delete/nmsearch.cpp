#include <cstdio>
#include <algorithm>
#include <math.h>
#include <time.h>
using namespace std;
void getNearestLauncher(int b,int n,float * xyz,int m,float * xyz2,float * result,int * result_i);
void getNearestLauncher2(int b,int n,float * xyz,int m,float * xyz2,float * result,int * result_i,float * result2,int * result2_i);
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
	int b=32;
	int n=16384;
	int m=16384;
	float * xyz=new float[b*n*3];
	float * xyz2=new float[b*m*3];
	float * dists=new float[b*n];
	int * best=new int[b*n];
	float * dists2=new float[b*m];
	int * best2=new int[b*m];
	for (int i=0;i<b*n*3;i++){
		xyz[i]=randomf();
	}
	for (int i=0;i<b*m*3;i++){
		xyz2[i]=randomf();
	}
	double bestt=1e100;
	for (int t=0;t<20;t++){
		double t0=get_time();
		//getNearestLauncher(b,n,xyz,m,xyz2,dists,best);
		//getNearestLauncher(b,m,xyz2,n,xyz,dists2,best2);
		getNearestLauncher2(b,n,xyz,m,xyz2,dists,best,dists2,best2);
		double total_t=get_time()-t0;
		bestt=min(bestt,total_t);
	}
	for (int i=0;i<100;i++){
		int j=rand()%b;
		int k=rand()%n;
		float bestd=0;
		int best_i=0;
		float x1=xyz[(j*n+k)*3+0];
		float y1=xyz[(j*n+k)*3+1];
		float z1=xyz[(j*n+k)*3+2];
		for (int l=0;l<m;l++){
			float x2=xyz2[(j*m+l)*3+0];
			float y2=xyz2[(j*m+l)*3+1];
			float z2=xyz2[(j*m+l)*3+2];
			float d=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
			if (l==0 || d<bestd){
				bestd=d;
				best_i=l;
			}
		}
		if (fabs(bestd-dists[(j*n+k)])>1e-6 || best_i!=best[(j*n+k)]){
			printf("WA j=%d k=%d best %f %f best_i %d %d\n",j,k,bestd,dists[j*n+k],best_i,best[j*n+k]);
		}
	}
	for (int i=0;i<100;i++){
		int j=rand()%b;
		int k=rand()%m;
		float bestd=0;
		int best_i=0;
		float x1=xyz2[(j*m+k)*3+0];
		float y1=xyz2[(j*m+k)*3+1];
		float z1=xyz2[(j*m+k)*3+2];
		for (int l=0;l<n;l++){
			float x2=xyz[(j*n+l)*3+0];
			float y2=xyz[(j*n+l)*3+1];
			float z2=xyz[(j*n+l)*3+2];
			float d=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
			if (l==0 || d<bestd){
				bestd=d;
				best_i=l;
			}
		}
		if (fabs(bestd-dists2[(j*m+k)])>1e-6 || best_i!=best2[(j*m+k)]){
			printf("WA2 j=%d k=%d best %f %f best_i %d %d\n",j,k,bestd,dists2[j*m+k],best_i,best2[j*m+k]);
		}
	}
	printf("best wall time %f\n",bestt);
	return 0;
}

