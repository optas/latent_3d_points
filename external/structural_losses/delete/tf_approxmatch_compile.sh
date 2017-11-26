set -e
if [ 'tf_approxmatch_g.cu.o' -ot 'tf_approxmatch_g.cu' ] ; then
	echo 'nvcc'
	/usr/local/cuda-7.5/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
fi
if [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch.cpp' ] || [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch_g.cu.o' ] ; then
	echo 'g++'
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-7.5/include  -L /usr/local/cuda-7.5/lib64/ -O2
fi
