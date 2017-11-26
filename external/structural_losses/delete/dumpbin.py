import numpy as np
import cPickle as pickle
np.random.seed(100)

data,pred,ptcloud,dists=pickle.load(open('../TensorFlow/depthestimate/ptcloud_cnn_32.pkl','rb'))
pred=np.concatenate([pred,np.random.randn(len(pred),1024-32,3).astype('float32')],axis=1)
fout=open('binpoints','wb')
fout.write(ptcloud.tostring())
fout.write(pred.tostring())
fout.close()
