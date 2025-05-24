#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 5>>>();

    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA device sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Hello World from CPU!\n");
    return 0;
}
