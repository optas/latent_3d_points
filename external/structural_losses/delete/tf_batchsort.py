import tensorflow as tf
from tensorflow.python.framework import ops
batch_sort_module=tf.load_op_library('./tf_batchsort_so.so')

def batch_sort(inp):
	'''
Sorts the number in a batched manner
input: inp: (batch_size,#numbers)  input numbers
output: out: (batch_size,#numbers)   sorted numbers
output: outi: (batch_size,#numbers)   index of numbers in the original array
	'''
	return batch_sort_module.batch_sort(inp)
@tf.RegisterShape('BatchSort')
def _batch_sort_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(2)
	return [shape1,shape1]
@tf.RegisterShape('BatchSortGrad')
def _batch_sort_grad_shape(op):
	shape1=op.inputs[0].get_shape().with_rank(2)
	return [shape1]
@ops.RegisterGradient('BatchSort')
def _batch_sort_grad(op,grad_out,grad_outi):
	outi=op.outputs[1]
	grad_inp=batch_sort_module.batch_sort_grad(grad_out,outi)
	return grad_inp

if __name__=='__main__':
	import numpy as np
	import time
	b=32
	n=16384
	np.random.seed(100)
	batch=(np.argsort(np.random.randn(b,n).astype('float32'),axis=1)*0.001).astype('float32')
	batch_sorted=np.sort(batch,axis=1)
	batch_idx=np.argsort(batch,axis=1)
	batch_sorted_g=np.random.randn(b,n).astype('float32')
	batch_g=batch.copy()
	for i in xrange(b):
		batch_g[i,batch_idx[i]]=batch_sorted_g[i]

	with tf.Session('') as sess:
		with tf.device('/gpu:1'):
			inp=tf.Variable(batch)
			coefs=tf.Variable(batch_sorted_g)
			out,outi=batch_sort(inp)
			loss=tf.reduce_sum(out*coefs)
			grad=tf.gradients(loss,[inp])[0]
		sess.run(tf.initialize_all_variables())
		besttime=1e38
		for i in xrange(20):
			t0=time.time()
			pred,predi,predg=sess.run([out,outi,grad])
			besttime=min(besttime,time.time()-t0)
		print 'time',besttime
		print np.abs(pred-batch_sorted).max(),(predi!=batch_idx).sum(),np.abs(batch_g-predg).max()
