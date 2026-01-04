cc -O3 -msse2 -msse3 -msse4 softmax.c -o softmax.out
./softmax.out

# nvcc -O3 ./flashattention_v1.cu -o flash_attn_test.out
# ./flash_attn_test.out
