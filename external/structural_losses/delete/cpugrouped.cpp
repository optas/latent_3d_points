#include <cstdio>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <vector>
using namespace std;
struct Point3fi{
	float x,y,z;
	int id;
};
struct CompareX{
	bool operator()(const Point3fi &a,const Point3fi &b)const{
		return a.x<b.x;
	}
};
struct CompareY{
	bool operator()(const Point3fi &a,const Point3fi &b)const{
		return a.y<b.y;
	}
};
struct CompareZ{
	bool operator()(const Point3fi &a,const Point3fi &b)const{
		return a.z<b.z;
	}
};
template<int batchsize>
void buildGroups(Point3fi * A,float * aabb,int beg,int end,int direction){
	if (end-beg<=batchsize){
		int sz=end-beg;
		if (sz==0)
			return;
		double mx=0,my=0,mz=0;
		double minx=0,maxx=0,miny=0,maxy=0,minz=0,maxz=0;
		for (int i=0;i<sz;i++){
			double x=A[beg+i].x;
			double y=A[beg+i].y;
			double z=A[beg+i].z;
			mx+=x;
			my+=y;
			mz+=z;
			if (i==0){
				minx=maxx=x;
				miny=maxy=y;
				minz=maxz=z;
			}else{
				minx=min(minx,x);
				maxx=max(maxx,x);
				miny=min(miny,y);
				maxy=max(maxy,y);
				minz=min(minz,z);
				maxz=max(maxz,z);
			}
		}
		aabb[(beg/batchsize)*6+0]=minx;
		aabb[(beg/batchsize)*6+1]=maxx;
		aabb[(beg/batchsize)*6+2]=miny;
		aabb[(beg/batchsize)*6+3]=maxy;
		aabb[(beg/batchsize)*6+4]=minz;
		aabb[(beg/batchsize)*6+5]=maxz;
		mx*=(1.0/sz);
		my*=(1.0/sz);
		mz*=(1.0/sz);
		double best=0;
		int besti=0;
		for (int i=0;i<sz;i++){
			double dx=A[beg].x-mx;
			double dy=A[beg].y-my;
			double dz=A[beg].z-mz;
			double d=(dx*dx+dy*dy+dz*dz);
			if (i==0 || d<best){
				best=d;
				besti=i;
			}
		}
		swap(A[beg+besti],A[beg]);
		return;
	}
	int m=(end-beg)/2;
	m=(m+batchsize-1)/batchsize*batchsize;
	m+=beg;
	if (direction==0){
		nth_element(A+beg,A+m,A+end,CompareX());
	}else if (direction==1){
		nth_element(A+beg,A+m,A+end,CompareY());
	}else{
		nth_element(A+beg,A+m,A+end,CompareZ());
	}
	buildGroups<batchsize>(A,aabb,beg,m,(direction+1)%3);
	buildGroups<batchsize>(A,aabb,m,end,(direction+1)%3);
}
static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}
template<int batchsize>
void getNearestNeighbor(int b,int n,float * xyz,int m,float * xyz2,float * result,int * result_i){
	vector<Point3fi> buffer(m);
	vector<float> buffer_aabb((m+(batchsize-1))/batchsize*6);
	vector<Point3fi> examples((m+(batchsize-1))/batchsize);
	int hitcnt=0;
	for (int i=0;i<b;i++){
		for (int j=0;j<m;j++){
			buffer[j].x=xyz2[(i*m+j)*3+0];
			buffer[j].y=xyz2[(i*m+j)*3+1];
			buffer[j].z=xyz2[(i*m+j)*3+2];
			buffer[j].id=j;
		}
		buildGroups<batchsize>(&(buffer[0]),&(buffer_aabb[0]),0,m,0);
		for (int k=0;k*batchsize<m;k++){
			examples[k]=buffer[k*batchsize];
		}
		for (int j=0;j<n;j++){
			double x1=xyz[(i*n+j)*3+0];
			double y1=xyz[(i*n+j)*3+1];
			double z1=xyz[(i*n+j)*3+2];
			double best=0;
			int best_i=0;
			for (int k=0;k*batchsize<m;k++){
				double x2=examples[k].x-x1;
				double y2=examples[k].y-y1;
				double z2=examples[k].z-z1;
				double d=x2*x2+y2*y2+z2*z2;
				if (k==0 || d<best){
					best=d;
					best_i=examples[k].id;
				}
			}
			for (int k=0;k*batchsize<m;k++){
				double x3=min(max(buffer_aabb[k*6+0],float(x1)),buffer_aabb[k*6+1])-x1;
				double y3=min(max(buffer_aabb[k*6+2],float(y1)),buffer_aabb[k*6+3])-y1;
				double z3=min(max(buffer_aabb[k*6+4],float(z1)),buffer_aabb[k*6+5])-z1;
				double d3=x3*x3+y3*y3+z3*z3;
				if (d3>=best)
					continue;
				hitcnt++;
				for (int l=1;l<min(batchsize,m-k*batchsize);l++){
					double x2=buffer[k*batchsize+l].x-x1;
					double y2=buffer[k*batchsize+l].y-y1;
					double z2=buffer[k*batchsize+l].z-z1;
					double d=x2*x2+y2*y2+z2*z2;
					if (d<best){
						best=d;
						best_i=buffer[k*batchsize+l].id;
					}
				}
			}
			result[i*n+j]=best;
			result_i[i*n+j]=best_i;
		}
	}
	printf("hitcnt %f\n",double(hitcnt)/(b*n));
}
float randomf(){
	return (rand()+0.5)/(RAND_MAX+1.0);
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
	FILE * fin=fopen("binpoints","rb");
	fread(xyz,3*4,b*n,fin);
	for (int i=0;i<b*m*3;i++){
		xyz2[i]=randomf();
	}
	//fread(xyz2,3*4,b*m,fin);
	double bestt=1e100;
	for (int t=0;t<3;t++){
		double t0=get_time();
		getNearestNeighbor<128>(b,n,xyz,m,xyz2,dists,best);
		getNearestNeighbor<128>(b,m,xyz2,n,xyz,dists2,best2);
		double total_t=get_time()-t0;
		printf("t=%d time=%f\n",t,total_t);
		bestt=min(bestt,total_t);
	}
	for (int i=0;i<100;i++){
		int j=rand()%b;
		int k=rand()%n;
		float bestd=dists[(j*n+k)];
		int best_i=best[(j*n+k)];
		float x1=xyz[(j*n+k)*3+0];
		float y1=xyz[(j*n+k)*3+1];
		float z1=xyz[(j*n+k)*3+2];
		for (int l=0;l<m;l++){
			float x2=xyz2[(j*m+l)*3+0];
			float y2=xyz2[(j*m+l)*3+1];
			float z2=xyz2[(j*m+l)*3+2];
			float d=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
			if (l==best_i){
				if (fabs(bestd-d)>1e-6){
					printf("WA j=%d k=%d best_i %d bestd %f %f mismatch\n",j,k,best_i,bestd,d);
					break;
				}
			}else if (d<bestd-1e-6){
				printf("WA j=%d k=%d best_i %d bestd %f l %d d %f worse\n",j,k,best_i,bestd,l,d);
				break;
			}
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

