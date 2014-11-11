#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>

using namespace std;

// const int NUM_BLOCKS = 4;
// const int THREADS_PER_BLOCK = 100;

// kernel which copies data from d_array to destinationArray  
__global__ void EvaluateFSM(int* d_fsm, int* threadResultStates, int* d_input, int* d_startStates, size_t pitch, int NUM_SYMBOLS, int NUM_STATES) {  
  int currentState = 0;

  for (int i = 0; i < 8; i++) { 
    int symbol = d_input[i];
    int* rowData = (int*) (((char*)d_fsm) + (currentState * pitch));
    currentState = rowData[symbol];
    threadResultStates[i] = currentState;
  }  
}


int main(int argc, char** argv) {  

	// FSM for detecting three ones in a row  
	const int NUM_STATES = 4; 
  const int NUM_SYMBOLS = 2;
  int fsm[NUM_STATES][NUM_SYMBOLS] = { {0, 1}, {0, 2}, {0, 3}, {3, 3} };
  int *d_fsm;

  int input[10] = { 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 };
  int *d_input;

  int startStates[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int *d_startStates;

  int h_threadResultStates[NUM_SYMBOLS*NUM_STATES] = { -1, -1, -1, -1, -1, -1, -1, -1 };
  int* d_threadResultStates;

  // The pitch value assigned by cudaMallocPitch  
  // (which ensures correct data structure alignment)  
  size_t pitch;   

	// Allocate the device memory
  cudaMallocPitch(&d_fsm, &pitch, NUM_SYMBOLS * sizeof(int), NUM_STATES);   
  cudaMalloc(&d_threadResultStates, (NUM_SYMBOLS * NUM_STATES * sizeof(int))); 
  cudaMalloc(&d_input, 10); 
  cudaMalloc(&d_startStates, 10);

  // Copy host memory to device memory
  cudaMemcpy2D(d_fsm, pitch, fsm, (NUM_SYMBOLS * sizeof(int)), (NUM_SYMBOLS * sizeof(int)), NUM_STATES, cudaMemcpyHostToDevice);  
  cudaMemcpy(d_threadResultStates, h_threadResultStates, NUM_STATES*NUM_SYMBOLS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input, 10, cudaMemcpyHostToDevice);
  cudaMemcpy(d_startStates, startStates, 10, cudaMemcpyHostToDevice);

  // Call the kernel
  EvaluateFSM<<<100, 512>>>(d_fsm, d_threadResultStates, d_input, d_startStates, pitch, NUM_SYMBOLS, NUM_STATES);

  // Copy the data back to the host memory  
  cudaMemcpy(h_threadResultStates, d_threadResultStates, (NUM_SYMBOLS * NUM_STATES * sizeof(int)), cudaMemcpyDeviceToHost);  

  // Print out the values (all the values are 123)  
  for (int i = 0 ; i < NUM_STATES * NUM_SYMBOLS; i++) { 
      cout << "h_threadResultStates[" << i << "] = " << h_threadResultStates[i] << endl;   
  }  
}  