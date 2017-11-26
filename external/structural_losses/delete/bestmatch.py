import ctypes as ct
import numpy as np

dll=np.ctypeslib.load_library('bestmatch_so','.')

def bestmatch(cost):
	n,m=cost.shape
	assert n<=m
	cost=np.require(cost,'float64','C')
	match=np.empty((m,),dtype='int32')
	ret=np.empty((1,),dtype='float64')
	dll.bestmatch(
		ct.c_int(n),
		ct.c_int(m),
		cost.ctypes.data_as(ct.c_void_p),
		match.ctypes.data_as(ct.c_void_p),
		ret.ctypes.data_as(ct.c_void_p),
	)
	ret=float(ret[0])
	return ret,match

if __name__=='__main__':
	#np.random.seed(100)
	#a=np.random.rand(100,200)
	#print a
	#val,match=bestmatch(a)
	#print val,sum(a[match[i],i] for i in xrange(len(match)) if match[i]!=-1)
	import time
	t0=time.time()
	u=np.random.rand(1024,3)
	v=np.random.rand(1024,3)
	a=((u[:,None,:]-v[None,:,:])**2).sum(axis=-1)
	val,match=bestmatch(a)
	print val,match,time.time()-t0
