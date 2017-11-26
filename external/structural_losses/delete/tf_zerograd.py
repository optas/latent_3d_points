import tensorflow as tf
from tensorflow.python.framework import ops
zerograd_module=tf.load_op_library('./tf_zerograd_so.so')

def zero_grad(inp):
	return zerograd_module.zero_grad(inp)
ops.NoGradient('ZeroGrad')
@tf.RegisterShape('ZeroGrad')
def _zero_grad_shape(op):
	shape1=op.inputs[0].get_shape()
	return [shape1]

if __name__=='__main__':
	import numpy as np
	a=np.random.randn(32,100,3).astype('float32')

	with tf.device('/gpu:0'):
		inp=tf.Variable(a)
		out=zero_grad(inp)
		grad=tf.gradients(tf.reduce_sum(out)+tf.reduce_sum(inp),[inp])[0]
	with tf.Session('') as sess:
		sess.run(tf.initialize_all_variables())
		ret1,ret2=sess.run([out,grad])
	print np.abs(ret1-a).max()
	print np.abs(ret2-1).max()
