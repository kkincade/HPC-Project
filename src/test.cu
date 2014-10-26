#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <map>
#include <thrust/host_vector.h>
 
// Kernel that executes on the CUDA device
__global__ void square_array(int *a, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = idx;
}
 
// main routine that executes on the host
int main(void) {

	int *thread_ids_h, *thread_ids_d;  // Pointer to host & device arrays
	const int N = 10;  // Number of elements in arrays
	size_t size = N * sizeof(int);

	thread_ids_h = (int *)malloc(size); // Allocate array on host

	cudaMalloc((void **) &thread_ids_d, size); // Allocate array on device

	// Initialize host array and copy it to CUDA device
	for (int i=0; i<N; i++) thread_ids_h[i] = (int)i;

	cudaMemcpy(thread_ids_d, thread_ids_h, size, cudaMemcpyHostToDevice);

	// Do calculation on device:
	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	square_array <<< n_blocks, block_size >>> (thread_ids_d, N);

	// Retrieve result from device and store it in host array
	cudaMemcpy(thread_ids_h, thread_ids_d, size, cudaMemcpyDeviceToHost);

	// Print results
	for (int i=0; i<N; i++) printf("%i %i\n", i, thread_ids_h[i]);

	// Cleanup
	free(thread_ids_h); cudaFree(thread_ids_d);
}

// using namespace std;

// #define NUM_BLOCKS 1
// #define NUM_THREADS 256

// // 1. Access the FSM from global memory. 
// // 2. Use the ThreadID to figure out the start state and the input characters. 
// // 3. Evaulate the FSM.
// // 4. Save resulting state in global memory.
// __global__ static void evaluate_fsm(int *threadIDs[], int length) {

// 	// Get thread and block IDs
// 	// const int tid = threadIdx.x;
//     // const int bid = blockIdx.x;



// 	// for (int i = 0; i < NUM_THREADS; i++) {
// 	// 	if ((*threadIDs)[i] == -1) {
// 	// 		(*threadIDs)[i] = tid;
// 	// 	}
// 	// }
    