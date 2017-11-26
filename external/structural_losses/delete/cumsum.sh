set -e
if [ 'cumsum.cu.o' -ot 'cumsum.cu' ]; then
	echo 'compiling cuda'
	/usr/local/cuda-7.5/bin/nvcc cumsum.cu -o cumsum.cu.o -c -O2
fi
if [ 'cumsum' -ot 'cumsum.cpp' ] || [ 'cumsum' -ot 'cumsum.cu.o' ]; then
	echo 'compiling cpp'
	g++ cumsum.cpp -o cumsum cumsum.cu.o -O2 -Wall -L /usr/local/cuda-7.5/lib64 -I /usr/local/cuda-7.5/include -lcudart
fi
