cc -O3 -msse2 -msse3 -msse4 softmax.c -o softmax.out -lm
./softmax.out

nvcc matrixMultiple.cu -O3 -o matrixMultiple -lcublas
./matrixMultiple
./matrixMultiple
