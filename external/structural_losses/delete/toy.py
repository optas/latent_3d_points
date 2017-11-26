import tensorflow as tf
from tensorflow.python.framework import ops
import tf_approxmatch
import tf_zerograd

if __name__=='__main__':
	alpha=0.5
	beta=2.0
	import bestmatch
	import numpy as np
	import math
	import random
	import cv2

	import tf_nndistance
	import tf_sampling

	npoint=100

	with tf.device('/gpu:3'):
		pt_in=tf.placeholder(tf.float32,shape=(1,npoint*4,3))
		mypoints=tf.Variable(np.random.randn(1,npoint,3).astype('float32'))
		#match=tf_approxmatch.approx_match(pt_in,mypoints)
		#loss=tf.reduce_sum(tf_approxmatch.match_cost(pt_in,mypoints,match))
		#distf,_,distb,_=tf_nndistance.nn_distance(pt_in,mypoints)
		#loss=tf.reduce_sum((distf+1e-9)**0.5)*0.5+tf.reduce_sum((distb+1e-9)**0.5)*0.5
		#loss+=tf.reduce_sum((distf+1e-9)**0.5)*0.5+tf.reduce_sum((distb+1e-9)**0.5)*0.5
		#loss=tf.reduce_sum((distf+1e-9)**0.5)*0.5+tf.reduce_sum((distb+1e-9)**0.5)*0.5
		#loss=tf.reduce_max((distf+1e-9)**0.5)*0.5*npoint+tf.reduce_max((distb+1e-9)**0.5)*0.5*npoint

		#distf,forward_i,distb,backward_i=tf_nndistance.nn_distance(pt_in,mypoints)
		#my_1=tf_sampling.gather_point(mypoints,forward_i)
		#_,_,_,my_1_i=tf_nndistance.nn_distance(pt_in,my_1)
		#pt_2=tf_sampling.gather_point(pt_in,my_1_i)
		#pt_1=tf_sampling.gather_point(pt_in,backward_i)
		#_,pt_1_i,_,_=tf_nndistance.nn_distance(pt_1,mypoints)
		#my_2=tf_sampling.gather_point(mypoints,pt_1_i)
		#loss=tf.reduce_sum(distf)+tf.reduce_sum(distb)+tf.reduce_sum((mypoints-my_2)**2)

		dists_forward_old,_,dists_backward,backward_i=tf_nndistance.nn_distance(pt_in,mypoints)
		projected=tf_sampling.gather_point(pt_in,backward_i)
		modified=mypoints+tf_zerograd.zero_grad(projected-mypoints)
		dists_forward,_,_,_=tf_nndistance.nn_distance(pt_in,modified)
		dists_forward=tf.reduce_sum(dists_forward)
		dists_backward=tf.reduce_sum(dists_backward)
		loss=dists_forward+dists_backward


		optimizer=tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
	with tf.Session('') as sess:
		sess.run(tf.initialize_all_variables())
		while True:
			meanloss=0
			meantrueloss=0
			for i in xrange(1001):
				phi=np.random.rand(4*npoint)*math.pi*2
				tpoints=(np.hstack([np.cos(phi)[:,None],np.sin(phi)[:,None],(phi*0)[:,None]])*random.random())[None,:,:]
				#tpoints=((np.random.rand(npoint*4)-0.5)[:,None]*[0,2,0]+[(random.random()-0.5)*2,0,0]).astype('float32')[None,:,:]
				#tpoints=np.hstack([np.linspace(-1,1,npoint*4)[:,None],(random.random()*2*np.linspace(1,0,npoint*4)**2)[:,None],np.zeros((npoint*4,1))])[None,:,:]
				trainloss,_=sess.run([loss,optimizer],feed_dict={pt_in:tpoints.astype('float32')})
			#trainloss,trainmatch=sess.run([loss,match],feed_dict={pt_in:tpoints.astype('float32')})
			trainloss=sess.run(loss,feed_dict={pt_in:tpoints.astype('float32')})
			show=np.zeros((400,400,3),dtype='uint8')^255
			trainmypoints=sess.run(mypoints)
			#for i in xrange(len(tpoints[0])):
				#u=np.random.choice(range(len(trainmypoints[0])),p=trainmatch[0].T[i])
				#cv2.line(show,
					#(int(tpoints[0][i,1]*100+200),int(tpoints[0][i,0]*100+200)),
					#(int(trainmypoints[0][u,1]*100+200),int(trainmypoints[0][u,0]*100+200)),
					#cv2.cv.CV_RGB(0,255,0))
			for x,y,z in tpoints[0]:
				cv2.circle(show,(int(y*100+200),int(x*100+200)),2,cv2.cv.CV_RGB(255,0,0))
			for x,y,z in trainmypoints[0]:
				cv2.circle(show,(int(y*100+200),int(x*100+200)),3,cv2.cv.CV_RGB(0,0,255))
			cost=((tpoints[0][:,None,:]-np.repeat(trainmypoints[0][None,:,:],4,axis=1))**2).sum(axis=2)**0.5
			#trueloss=bestmatch.bestmatch(cost)[0]
			print trainloss#,trueloss
			cv2.imshow('show',show)
			cmd=cv2.waitKey(10)%256
			if cmd==ord('q'):
				break
