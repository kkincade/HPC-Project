#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>

using namespace std;

const int NUM_BLOCKS = 1;
const int THREADS_PER_BLOCK = 17;
const int CHUNK_SIZE = 3;
const int NUM_STATES = 4; 
const int NUM_SYMBOLS = 2;

// kernel which copies data from d_array to destinationArray  
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, int* d_startStates, size_t pitch, int NUM_SYMBOLS, int NUM_STATES, int CHUNK_SIZE) {  
  int currentState = d_startStates[threadIdx.x];
  int* input = (int*) ((char*) d_inputs + (threadIdx.x * pitch));

  for (int i = 0; i < CHUNK_SIZE; i++) { 
    int symbol = input[i];
    int* rowData = (int*) ((char*) d_fsm + (currentState * pitch));
    currentState = rowData[symbol];
  }  

  d_threadResultStates[threadIdx.x] = currentState;
}


int main(int argc, char** argv) {  
  int inputString[12*15] = { 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                          0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1 };
  int inputIndex = 0;
  int currentState = 0;

  int inputLength = sizeof(inputString) / sizeof(int);

  // FSM for detecting three ones in a row
  int fsm[NUM_STATES][NUM_SYMBOLS] = { {0, 1}, {0, 2}, {0, 3}, {3, 3} };
  int *d_fsm;

  int inputs[THREADS_PER_BLOCK][CHUNK_SIZE];
  int *d_inputs;

  int startStates[THREADS_PER_BLOCK];
  fill_n(startStates, THREADS_PER_BLOCK, -1);
  int *d_startStates;

  int h_threadResultStates[THREADS_PER_BLOCK];
  fill_n(h_threadResultStates, THREADS_PER_BLOCK, -1);
  int* d_threadResultStates;

  // ------------------------- MAIN EXECUTION LOOP ----------------------------
  int threadIndex;
  int chunk[CHUNK_SIZE];
  bool loadingChunks;
  bool emptyChunk;

  while (inputIndex < inputLength) {

    cout << "Loop!!" << endl;

    threadIndex = 0;
    loadingChunks = true;
    emptyChunk = false;    

    while (loadingChunks && threadIndex < THREADS_PER_BLOCK) {
      // Populate the chunk
      for (int j = 0; j < CHUNK_SIZE; j += 1) {
        // TODO: Handle cases where the length of inputString is not divisible by CHUNK_SIZE
        if (inputIndex < inputLength) {
          chunk[j] = inputString[inputIndex];
          inputIndex += 1;
        } else {
          // Reached end of input string
          loadingChunks = false;
          // Set flag if chunk is empty so it doesn't get assigned to a thread
          if (j == 0) {
            emptyChunk = true;
          }
          break;
        }
      }

      if (emptyChunk) break;

      if (threadIndex == 0) {
        // Only copy once for first thread
        for (int k = 0; k < CHUNK_SIZE; k++) {
          inputs[threadIndex][k] = chunk[k];
        }
        
        // Add start state for thread
        startStates[threadIndex] = currentState;

        threadIndex += 1;
      } else {
        // Copy chunk to inputs for every possible start state
        for (int j = 0; j < NUM_STATES; j += 1) {
          if (threadIndex < THREADS_PER_BLOCK) {
            // Add input for thread
            for (int k = 0; k < CHUNK_SIZE; k++) {
              inputs[threadIndex][k] = chunk[k];
            }

            // Add start state for thread
            startStates[threadIndex] = j;

            threadIndex += 1;
          } else {
            // TODO: remember where we were for next iteration
            // All threads have been initialized
            loadingChunks = false;
            break;
          }
        }
      }
    }

    // The pitch value assigned by cudaMallocPitch  
    // (which ensures correct data structure alignment)  
    size_t pitch;   

    // Allocate the device memory
    cudaMallocPitch(&d_fsm, &pitch, sizeof(int)*NUM_SYMBOLS, NUM_STATES);
    cudaMallocPitch(&d_inputs, &pitch, sizeof(int)*CHUNK_SIZE, THREADS_PER_BLOCK);  
    cudaMalloc(&d_threadResultStates, sizeof(int)*THREADS_PER_BLOCK); 
    cudaMalloc(&d_startStates, sizeof(int)*THREADS_PER_BLOCK);

    // Copy host memory to device memory
    cudaMemcpy2D(d_fsm, pitch, fsm, (NUM_SYMBOLS * sizeof(int)), (NUM_SYMBOLS * sizeof(int)), NUM_STATES, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_inputs, pitch, inputs, (CHUNK_SIZE * sizeof(int)), (CHUNK_SIZE * sizeof(int)), THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_threadResultStates, h_threadResultStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startStates, startStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyHostToDevice);

    // Call the kernel
    EvaluateFSM<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_fsm, d_threadResultStates, d_inputs, d_startStates, pitch, NUM_SYMBOLS, NUM_STATES, CHUNK_SIZE);

    // Copy the data back to the host memory  
    cudaMemcpy(h_threadResultStates, d_threadResultStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyDeviceToHost);  

    // Find the current state by using the thread results
    currentState = h_threadResultStates[0];

    cout << "Level 0: " << currentState << endl;

    int predictionLevel = 0;
    while ((1 + (predictionLevel*NUM_STATES) + currentState) < ( sizeof(h_threadResultStates) / sizeof(int) )) {
      currentState = h_threadResultStates[1 + (predictionLevel*NUM_STATES) + currentState];
      predictionLevel += 1;
      cout << "Level " << predictionLevel << ": " << currentState << endl;
    }

    cout << "Current State: " << currentState << endl;
    cout << "Input Index: " << inputIndex << endl;
  }

  cout << "Final State: " << currentState << endl;

}