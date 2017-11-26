if [ 'tf_zerograd_so.so' -ot 'tf_zerograd.cpp' ] ; then
	echo 'g++'
	g++ -std=c++11 tf_zerograd.cpp -o tf_zerograd_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-7.5/include -lcudart -L /usr/local/cuda-7.5/lib64/ -O2
fi
