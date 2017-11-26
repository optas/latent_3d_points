set -e
if [ 'approxmatch2.cu.o' -ot 'approxmatch2.cu' ]; then
	echo 'compiling cuda'
	/usr/local/cuda-7.5/bin/nvcc approxmatch2.cu -o approxmatch2.cu.o -c -O2
fi
if [ 'approxmatch2' -ot 'approxmatch2.cpp' ] || [ 'approxmatch2' -ot 'approxmatch2.cu.o' ]; then
	echo 'compiling cpp'
	g++ approxmatch2.cpp -o approxmatch2 approxmatch2.cu.o -O2 -Wall -L /usr/local/cuda-7.5/lib64 -I /usr/local/cuda-7.5/include -lcudart
fi
