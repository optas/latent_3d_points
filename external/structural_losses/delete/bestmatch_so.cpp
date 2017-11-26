#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
using namespace std;

extern "C"{

void bestmatch(int n,int m,double * A,int * match,double * ret){
	static const double tolerance=1e-7;
	struct KMSolveContext{
		vector<double> lx,ly,slack;
		vector<int> vx,vy;
		double * A;
		int * match,n,m;
		bool dfs_search(int u){
			//printf("dfs %d match[0]=%d\n",u,match[0]);
			vx[u]=1;
			for (int i=0;i<m;i++){
				if (!vy[i]){
					double t=A[u*m+i]-lx[u]-ly[i];
					if (t<tolerance){
						vy[i]=1;
						if (match[i]==-1 || dfs_search(match[i])){
							//printf("match %d = %d\n",i,u);
							match[i]=u;
							return true;
						}
					}else{
						slack[i]=min(slack[i],t);
					}
				}
			}
			return false;
		}
		KMSolveContext(int n,int m,double *A,int * match,double * ret){
			lx.resize(n);
			ly.assign(m,0);
			slack.resize(m);
			vx.resize(n);
			vy.resize(m);
			this->A=A;
			this->n=n;
			this->m=m;
			this->match=match;
			for (int i=0;i<n;i++){
				lx[i]=*min_element(A+i*m,A+i*m+m);
			}
			double big=0;
			for (int i=0;i<n*m;i++)
				big=max(big,fabs(A[i]));
			big=big*4+100;
			for (int i=0;i<m;i++)
				match[i]=-1;
			int t=0;
			for (int i=0;i<n;i++){
				//printf("i=%d\n",i);
				for (int j=0;j<m;j++){
					slack[j]=big;
				}
				while (true){
					t++;
					//if (t>=20)
						//break;
					//for (int j=0;j<m;j++)
						//printf("match %d=%d\n",j,match[j]);
					//for (int j=0;j<n;j++)
						//printf("lx %d=%f\n",j,lx[j]);
					//for (int j=0;j<m;j++)
						//printf("ly %d=%f\n",j,ly[j]);
					for (int k=0;k<n;k++)
						vx[k]=0;
					for (int k=0;k<m;k++)
						vy[k]=0;
					if (dfs_search(i)){
						//puts("return true");
						break;
					}
					double d=big;
					for (int j=0;j<m;j++)
						if (!vy[j] && d>slack[j])
							d=slack[j];
					for (int j=0;j<n;j++)
						if (vx[j])
							lx[j]+=d;
					for (int j=0;j<n;j++){
						if (vy[j])
							ly[j]-=d;
					}
				}
			}
			ret[0]=0;
			for (int i=0;i<m;i++)
				if (match[i]!=-1)
					ret[0]+=A[match[i]*m+i];
		}
	};
	KMSolveContext(n,m,A,match,ret);
}

}//extern "C"
