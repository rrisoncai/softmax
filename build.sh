g++ -O3 -march=native -std=c++17 softmax_avx_cpu.cpp -o softmax_avx_cpu
./softmax_avx_cpu

nvcc -O3 ./flashattention_v1.cu -o flash_attn_test
./flash_attn_test
