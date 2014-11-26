#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <sys/time.h>

using namespace std;

// Constants
const int NUM_BLOCKS = 1;
const int THREADS_PER_BLOCK = 1000;
const int CHUNK_SIZE = 1000;
const int NUM_STATES = 5;
const int NUM_SYMBOLS = 2;

// Function prototypes
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, int* d_startStates, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex);
void EvalueFSMonGPU(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void EvaluateSerialFSM(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void CreateInputString(int result[], int symbols[], long long int inputLength);


int main(int argc, char** argv) {

  // FSM for detecting three ones in a row
  int fsm[NUM_STATES][NUM_SYMBOLS] = { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {4, 4} };
  int symbols[NUM_SYMBOLS] = { 0, 1 };
  int currentState = 0;

  // Create input string
  srand(time(NULL));
  long long int inputLength = (long long int) 180000;
  int* inputString = new int[inputLength];
  CreateInputString(inputString, symbols, inputLength);

  struct timeval start, end;
  gettimeofday(&start, NULL);

  if (argc > 1 && string(argv[1]) == "serial") {
    EvaluateSerialFSM(fsm, inputString, currentState, inputLength);
  } else {
    EvalueFSMonGPU(fsm, inputString, currentState, inputLength);
  }

  gettimeofday(&end, NULL);
  double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  cout << "Time: " << delta << endl;

}


// Kernel method 
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, int* d_startStates, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex) {  
  // Make sure we don't go past the threads that have been assigned input
  if (threadIdx.x < maxThreadIndex) {
    int currentState = d_startStates[threadIdx.x];
    int* input = (int*) ((char*) d_inputs + (threadIdx.x * inputsPitch));

    for (int i = 0; i < CHUNK_SIZE; i++) { 

      // Make sure there is still valid input left in the chunk
      if (input[i] == -1) break;

      int symbol = input[i];
      int* rowData = (int*) ((char*) d_fsm + (currentState * fsmPitch));
      currentState = rowData[symbol];
    }  

    d_threadResultStates[threadIdx.x] = currentState;
  } else {
    // Idle thread, just store -1 as the result state
    d_threadResultStates[threadIdx.x] = -1;
  }
}


// Evaluates the FSM using the GPU by calling the kernel method
void EvalueFSMonGPU(int fsm[][NUM_SYMBOLS], int inputString[], int currentState, long long int inputLength) {
  // ------------------------- PARALLEL EXECUTION LOOP ----------------------------
  long long int inputIndex = 0;
  int *d_fsm;

  // TODO: Make sure this way doesn't break our code!!!
  // int inputs[THREADS_PER_BLOCK][CHUNK_SIZE];
  int** inputs = new int*[THREADS_PER_BLOCK];
  for (int i = 0; i < THREADS_PER_BLOCK; i += 1) { inputs[i] = new int[CHUNK_SIZE]; }
  int *d_inputs;

  int startStates[THREADS_PER_BLOCK];
  fill_n(startStates, THREADS_PER_BLOCK, -1);
  int *d_startStates;

  int h_threadResultStates[THREADS_PER_BLOCK];
  fill_n(h_threadResultStates, THREADS_PER_BLOCK, -1);
  int* d_threadResultStates;

  int threadIndex;
  int chunk[CHUNK_SIZE];
  bool loadingChunks;
  bool emptyChunk;
  
  while (inputIndex < inputLength) {

    threadIndex = 0;
    loadingChunks = true;
    emptyChunk = false;

    // Fill all values with -1 so we can tell meaningful values from non-meaningful values
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
      fill_n(inputs[i], CHUNK_SIZE, -1);
    }

    // Chunkifying the input string 
    while (loadingChunks && threadIndex < THREADS_PER_BLOCK) {
      // Populate the chunk
      for (int j = 0; j < CHUNK_SIZE; j += 1) {
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

    // The pitch values assigned by cudaMallocPitch ensure correct data structure alignment
    size_t fsmPitch, inputsPitch;   

    // Allocate the device memory
    cudaMallocPitch(&d_fsm, &fsmPitch, sizeof(int)*NUM_SYMBOLS, NUM_STATES);
    cudaMallocPitch(&d_inputs, &inputsPitch, sizeof(int)*CHUNK_SIZE, THREADS_PER_BLOCK);  
    cudaMalloc(&d_threadResultStates, sizeof(int)*THREADS_PER_BLOCK); 
    cudaMalloc(&d_startStates, sizeof(int)*THREADS_PER_BLOCK);

    // Copy host memory to device memory
    cudaMemcpy2D(d_fsm, fsmPitch, fsm, (NUM_SYMBOLS * sizeof(int)), (NUM_SYMBOLS * sizeof(int)), NUM_STATES, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_inputs, inputsPitch, *inputs, (CHUNK_SIZE * sizeof(int)), (CHUNK_SIZE * sizeof(int)), THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_threadResultStates, h_threadResultStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startStates, startStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyHostToDevice);

    // Run FSM on GPU
    EvaluateFSM<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_fsm, d_threadResultStates, d_inputs, d_startStates, fsmPitch, inputsPitch, NUM_STATES, NUM_SYMBOLS, CHUNK_SIZE, threadIndex);

    // Copy the data back to the host memory  
    cudaMemcpy(h_threadResultStates, d_threadResultStates, (THREADS_PER_BLOCK * sizeof(int)), cudaMemcpyDeviceToHost);  

    // Find the new current state by using the enumerated thread results 
    // to see if we were able to find a valid future state
    currentState = h_threadResultStates[0];
    int predictionLevel = 0;
    while ((1 + (predictionLevel*NUM_STATES) + currentState) < ( sizeof(h_threadResultStates) / sizeof(int) )) {
      if ((h_threadResultStates[1 + (predictionLevel*NUM_STATES) + currentState]) != -1) {
        currentState = h_threadResultStates[1 + (predictionLevel*NUM_STATES) + currentState];
        predictionLevel += 1;
      } else {
        break;
      }
    }
  }

  cout << "Final State: " << currentState << endl;
}


// Evaluates the input string on the FSM serially as a benchmark for comparing our GPU implementation to
void EvaluateSerialFSM(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength) {
  int currentState = startState;
  for (long long int i = 0; i < inputLength; i++) {
    int symbol = inputString[i];
    currentState = fsm[currentState][symbol];
  }
  cout << "Serial FSM final state: " << currentState << endl;
}


// Creates a random input string of given length to use as the FSM's input
void CreateInputString(int result[], int symbols[], long long int inputLength) {
  for (long long int i = 0; i < inputLength; i++) {
    result[i] = symbols[rand() % NUM_SYMBOLS];
  }
}