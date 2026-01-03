g++ -O3 -march=native -std=c++17 softmax_avx_cpu.cpp -o softmax_avx_cpu.out
./softmax_avx_cpu.out

nvcc -O3 ./flashattention_v1.cu -o flash_attn_test.out
./flash_attn_test.out
