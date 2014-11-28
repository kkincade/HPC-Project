#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <stdio.h>
#include <sys/time.h>
#include "cuPrintf.cu"

using namespace std;

// Constants
const int NUM_BLOCKS = 1;
const int THREADS_PER_BLOCK = 1024;
const int NUM_THREADS = NUM_BLOCKS*THREADS_PER_BLOCK;
const int CHUNK_SIZE = 100;
const int NUM_STATES = 5;
const int NUM_SYMBOLS = 2;
const int NUM_INPUTS = 1 + ((NUM_THREADS - 1) / NUM_STATES);

// Textures
texture<int, cudaTextureType2D, cudaReadModeElementType> textureFSM;

// Function prototypes
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex, int hostCurrentState);
void EvaluateFSMonGPU(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void EvaluateSerialFSM(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void CreateInputString(int result[], int symbols[], long long int inputLength);


int main(int argc, char** argv) {

  // FSM for detecting three ones in a row
  int fsm[NUM_STATES][NUM_SYMBOLS] = { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {4, 4} };
  int symbols[NUM_SYMBOLS] = { 0, 1 };
  int currentState = 0;

  // Create input string
  srand(time(NULL));
  long long int inputLength = (long long int) 180000000;
  int* inputString = new int[inputLength];
  CreateInputString(inputString, symbols, inputLength);

  struct timeval start, end;
  gettimeofday(&start, NULL);

  if (argc > 1 && string(argv[1]) == "serial") {
    EvaluateSerialFSM(fsm, inputString, currentState, inputLength);
  } else {
    EvaluateFSMonGPU(fsm, inputString, currentState, inputLength);
  }

  gettimeofday(&end, NULL);
  double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  cout << "Time: " << delta << endl;

}


// Kernel method 
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex, int hostCurrentState) {  
  // Make sure we don't go past the threads that have been assigned input
  int threadID = threadIdx.x + (blockIdx.x * blockDim.x);

  if (threadID < maxThreadIndex) {
    // int currentState = d_startStates[threadID];
    int currentState;

    int* input;

    if (threadID == 0) {
      // First thread is running the actual current input (only thread using d_inputs[0])
      input = (int*) ((char*) d_inputs + (threadID * inputsPitch));
      currentState = hostCurrentState;
    } else {
      input = (int*) ((char*) d_inputs + (((threadID / NUM_STATES) + 1) * inputsPitch));
      currentState = ((threadID - 1) % NUM_STATES);
    }
    // int* input = (int*) ((char*) d_inputs + (threadID * inputsPitch));
    
    for (int i = 0; i < CHUNK_SIZE; i++) { 

      // Make sure there is still valid input left in the chunk
      if (input[i] == -1) break;

      int symbol = input[i];

      // The tex2D method thinks of the array as an "image", with
      // (x, y) coordinates, with the x value being the first argument.
      // Therefore we must switch our symbol and currentState when accessing.
      currentState = tex2D(textureFSM, symbol, currentState);
      
      // int* rowData = (int*) ((char*) d_fsm + (currentState * fsmPitch));
      // currentState = rowData[symbol];
    }  

    d_threadResultStates[threadID] = currentState;
  } else {
    // Idle thread, just store -1 as the result state
    d_threadResultStates[threadID] = -1;
  }
}


// Evaluates the FSM using the GPU by calling the kernel method
void EvaluateFSMonGPU(int fsm[][NUM_SYMBOLS], int inputString[], int currentState, long long int inputLength) {
  // cudaPrintfInit(); // Printf capabilities

  long long int inputStringIndex = 0;
  int *d_fsm;

  // int inputs[NUM_THREADS][CHUNK_SIZE];
  // Allocates the 2D array on the heap so we don't hit a seg fault
  // int** inputs = new int*[NUM_THREADS];
  int** inputs = new int*[NUM_INPUTS];
  for (int i = 0; i < NUM_INPUTS; i += 1) { inputs[i] = new int[CHUNK_SIZE]; }
  int *d_inputs;

  // int startStates[NUM_THREADS];
  // fill_n(startStates, NUM_THREADS, -1);
  // int *d_startStates;

  int h_threadResultStates[NUM_THREADS];
  fill_n(h_threadResultStates, NUM_THREADS, -1);
  int* d_threadResultStates;

  int threadIndex;
  bool loadingChunks;
  bool emptyChunk;

  while (inputStringIndex < inputLength) {

    threadIndex = 0;
    loadingChunks = true;
    emptyChunk = false;

    cout << "CHUNKIFY!!" << endl;

    // Chunkifying the input string 
    while (loadingChunks && threadIndex < NUM_THREADS) {
      int chunkStartIndex = inputStringIndex;

      // Find the input chunk indices
      for (int j = 0; j < CHUNK_SIZE; j += 1) {
        if (inputStringIndex < inputLength) {
          inputStringIndex += 1;
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
        for (int k = 0; k < (inputStringIndex - chunkStartIndex); k++) {
          // TODO: This should not be threadIndex!!!
          // HERE!!!
          // Just changed inputIndex to inputStringIndex. We need to make an inputIndex which
          // we can use instead of threadIndex.
          inputs[threadIndex][k] = inputString[chunkStartIndex + k];
        }
        
        // Add start state for thread
        // startStates[threadIndex] = currentState;

        threadIndex += 1;
      } else {
        // Copy inputString chunk for every possible start state

        // Add input for thread
        for (int j = 0; j < (inputStringIndex - chunkStartIndex); j++) {
          inputs[threadIndex][j] = inputString[chunkStartIndex + j];
        }

        threadIndex += NUM_STATES;

        if (threadIndex >= NUM_THREADS) {
          threadIndex = NUM_THREADS;
          loadingChunks = false;
        }

        // for (int j = 0; j < NUM_STATES; j += 1) {
        //   if (threadIndex < NUM_THREADS) {
        //     // Add input for thread
        //     for (int k = 0; k < (inputStringIndex - chunkStartIndex); k++) {
        //       inputs[threadIndex][k] = inputString[chunkStartIndex + k];
        //     }

        //     // Add start state for thread
        //     startStates[threadIndex] = j;

        //     threadIndex += 1;
        //   } else {
        //     // TODO: remember where we were for next iteration
        //     // All threads have been initialized
        //     loadingChunks = false;
        //     break;
        //   }
        // }
      }
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
      for (int j = 0; j < CHUNK_SIZE; j++) {
        cout << inputs[i][j];
      }
      cout << endl;
    }

    // The pitch values assigned by cudaMallocPitch ensure correct data structure alignment
    // size_t fsmPitch, inputsPitch;   

    // // Allocate the device memory
    // cudaMallocPitch(&d_fsm, &fsmPitch, sizeof(int)*NUM_SYMBOLS, NUM_STATES);
    // cudaMallocPitch(&d_inputs, &inputsPitch, sizeof(int)*CHUNK_SIZE, NUM_THREADS);  
    // cudaMalloc(&d_threadResultStates, sizeof(int)*NUM_THREADS); 
    // // cudaMalloc(&d_startStates, sizeof(int)*NUM_THREADS);

    // // Copy host memory to device memory
    // cudaMemcpy2D(d_fsm, fsmPitch, fsm, (NUM_SYMBOLS * sizeof(int)), (NUM_SYMBOLS * sizeof(int)), NUM_STATES, cudaMemcpyHostToDevice);
    // // cudaMemcpy2D(d_inputs, inputsPitch, *inputs, (CHUNK_SIZE * sizeof(int)), (CHUNK_SIZE * sizeof(int)), NUM_THREADS, cudaMemcpyHostToDevice);
    // cudaMemcpy2D(d_inputs, inputsPitch, *inputs, (CHUNK_SIZE * sizeof(int)), (CHUNK_SIZE * sizeof(int)), NUM_INPUTS, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_threadResultStates, h_threadResultStates, (NUM_THREADS * sizeof(int)), cudaMemcpyHostToDevice);
    // // cudaMemcpy(d_startStates, startStates, (NUM_THREADS * sizeof(int)), cudaMemcpyHostToDevice);

    // // Bind Texture to GPU Cache
    // static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    // cudaBindTexture2D(0, textureFSM, d_fsm, channelDesc, NUM_SYMBOLS, NUM_STATES, fsmPitch);
    // cudaFree(d_fsm);

    // // Run FSM on GPU
    // EvaluateFSM<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_fsm, d_threadResultStates, d_inputs, fsmPitch, inputsPitch, NUM_STATES, NUM_SYMBOLS, CHUNK_SIZE, threadIndex, currentState);
    // // cudaPrintfDisplay(stdout, true); // Printf capabilities

    // // Copy the data back to the host memory  
    // cudaMemcpy(h_threadResultStates, d_threadResultStates, (NUM_THREADS * sizeof(int)), cudaMemcpyDeviceToHost);  

    // // Find the new current state by using the enumerated thread results 
    // // to see if we were able to find a valid future state
    // currentState = h_threadResultStates[0];
    // int predictionLevel = 0;
    // while ((1 + (predictionLevel*NUM_STATES) + currentState) < ( sizeof(h_threadResultStates) / sizeof(int) )) {
    //   if ((1 + (predictionLevel*NUM_STATES) + currentState) < threadIndex) {
    //     currentState = h_threadResultStates[1 + (predictionLevel*NUM_STATES) + currentState];
    //     predictionLevel += 1;
    //   } else {
    //     break;
    //   }
    // }
  }

  cout << "GPU Final State: " << currentState << endl;

  // cudaPrintfEnd(); // Printf capabilities
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