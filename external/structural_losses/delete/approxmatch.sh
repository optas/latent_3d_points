set -e
if [ 'approxmatch.cu.o' -ot 'approxmatch.cu' ]; then
	echo 'compiling cuda'
	/usr/local/cuda-7.5/bin/nvcc approxmatch.cu -o approxmatch.cu.o -c -O2
fi
if [ 'approxmatch' -ot 'approxmatch.cpp' ] || [ 'approxmatch' -ot 'approxmatch.cu.o' ]; then
	echo 'compiling cpp'
	g++ approxmatch.cpp -o approxmatch approxmatch.cu.o -O2 -Wall -L /usr/local/cuda-7.5/lib64 -I /usr/local/cuda-7.5/include -lcudart
fi
