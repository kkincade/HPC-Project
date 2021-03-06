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
const int CHUNK_SIZE = 10000;
const int NUM_STATES = 64;
const int NUM_SYMBOLS = 4;
const int NUM_INPUTS = 2 + ((NUM_THREADS - 1) / NUM_STATES);

// Textures
texture<int, cudaTextureType2D, cudaReadModeElementType> textureFSM;

// Function prototypes
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex, int hostCurrentState);
void EvaluateFSMonGPU(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void EvaluateSerialFSM(int fsm[][NUM_SYMBOLS], int inputString[], int startState, long long int inputLength);
void CreateInputString(int result[], int symbols[], long long int inputLength);


int main(int argc, char** argv) {

  // FSM for detecting three ones in a row
  // 4 State FSM
  // int fsm[NUM_STATES][NUM_SYMBOLS] = { {1, 0, 0, 0}, {1, 0, 3, 1}, {0, 3, 2, 2}, {3, 3, 3, 2} };

  // 16 State FSM
  int fsm[NUM_STATES][NUM_SYMBOLS] = { {1, 0, 12, 0}, {1, 0, 3, 1}, {0, 3, 2, 2}, {4, 3, 8, 2}, 
                                       {1, 3, 5, 10}, {11, 0, 9, 1}, {0, 3, 8, 8}, {7, 10, 12, 7},
                                       {14, 3, 1, 5}, {4, 4, 4, 2}, {10, 13, 3, 4}, {5, 6, 7, 8},
                                       {0, 9, 0, 9}, {1, 7, 15, 1}, {2, 3, 2, 2}, {13, 14, 15, 2}
                                     };

  // 64 State FSM
  // int fsm[NUM_STATES][NUM_SYMBOLS] = { {1, 63, 12, 49}, {48, 0, 61, 1}, {57, 29, 47, 2}, {54, 46, 8, 2}, 
  //                                      {1, 42, 33, 10}, {60, 0, 9, 43}, {44, 3, 56, 8}, {45, 10, 53, 7},
  //                                      {62, 41, 1, 5}, {40, 4, 4, 59}, {55, 13, 39, 4}, {5, 38, 52, 8},
  //                                      {51, 9, 33, 9}, {58, 34, 15, 1}, {2, 35, 2, 36}, {13, 37, 15, 2},

  //                                      {1, 32, 12, 50}, {46, 31, 3, 1}, {41, 3, 2, 2}, {4, 13, 37, 12}, 
  //                                      {1, 49, 5, 10}, {11, 44, 30, 1}, {40, 28, 27, 26}, {7, 10, 12, 36},
  //                                      {48, 3, 1, 5}, {4, 4, 43, 2}, {10, 13, 39, 25}, {14, 6, 35, 8},
  //                                      {0, 20, 47, 21}, {22, 42, 15, 1}, {23, 38, 24, 2}, {34, 14, 15, 2},

  //                                      {19, 33, 12, 0}, {1, 0, 29, 1}, {17, 25, 16, 2}, {15, 21, 8, 2}, 
  //                                      {1, 32, 5, 10}, {11, 28, 18, 1}, {24, 3, 8, 8}, {7, 20, 12, 7},
  //                                      {31, 3, 1, 5}, {4, 4, 27, 2}, {10, 13, 23, 4}, {19, 6, 7, 8},
  //                                      {30, 9, 0, 9}, {1, 26, 15, 1}, {2, 3, 22, 2}, {13, 18, 15, 2},

  //                                      {1, 0, 17, 0}, {1, 13, 3, 1}, {0, 41, 63, 2}, {4, 58, 11, 2}, 
  //                                      {1, 16, 5, 10}, {11, 0, 12, 1}, {0, 62, 8, 8}, {56, 10, 57, 7},
  //                                      {14, 3, 15, 5}, {4, 11, 4, 2}, {61, 13, 3, 4}, {5, 55, 54, 53},
  //                                      {0, 14, 0, 9}, {10, 7, 15, 1}, {59, 3, 60, 2}, {50, 51, 15, 52}
  //                                    };

  int symbols[NUM_SYMBOLS] = { 0, 1, 2, 3 };
  int currentState = 0;

  // Create input string
  srand(time(NULL));

  // Set the input size
  long long int inputLength = (long long int) 200000000; 
  long long int inputLength = (long long int) 2000000000; 
  long long int inputLength = (long long int) 20000000000; 
  int* inputString = new int[inputLength];
  CreateInputString(inputString, symbols, inputLength);

  struct timeval start, end;

  if (argc > 1 && (string(argv[1]) == "serial" || string(argv[1]) == "both")) {
    cout << "---------- SERIAL IMPLEMENTATION ----------" << endl;
    gettimeofday(&start, NULL);
    EvaluateSerialFSM(fsm, inputString, currentState, inputLength);
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Time: " << delta << endl;
  }

  if ((argc > 1 && (string(argv[1]) == "both") || argc <= 1)) {
    cout << "----------- GPU IMPLEMENTATION ------------" << endl;
    gettimeofday(&start, NULL);
    EvaluateFSMonGPU(fsm, inputString, currentState, inputLength);
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Time: " << delta << endl;
  }
}


// Kernel method 
__global__ void EvaluateFSM(int* d_fsm, int* d_threadResultStates, int* d_inputs, size_t fsmPitch, size_t inputsPitch, int NUM_STATES, int NUM_SYMBOLS, int CHUNK_SIZE, int maxThreadIndex, int hostCurrentState) {  
  // Make sure we don't go past the threads that have been assigned input
  int threadID = threadIdx.x + (blockIdx.x * blockDim.x);

  if (threadID < maxThreadIndex) {
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
    
    for (int i = 0; i < CHUNK_SIZE; i++) { 

      // Make sure there is still valid input left in the chunk
      if (input[i] == -1) break;

      int symbol = input[i];

      // The tex2D method thinks of the array as an "image", with
      // (x, y) coordinates, with the x value being the first argument.
      // Therefore we must switch our symbol and currentState when accessing.
      currentState = tex2D(textureFSM, symbol, currentState);
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

  // Allocates the 2D array on the heap so we don't hit a seg fault
  int** inputs = new int*[NUM_INPUTS];
  for (int i = 0; i < NUM_INPUTS; i += 1) { inputs[i] = new int[CHUNK_SIZE]; }
  int *d_inputs;

  int h_threadResultStates[NUM_THREADS];
  int* d_threadResultStates;

  int threadIndex;
  int inputIndex;
  bool loadingChunks;
  bool emptyChunk;
  int predictionLevel;
  int chunkStartIndex;

  double kernelTime = 0.0;
  struct timeval kernelStart, kernelEnd;

  // The pitch values assigned by cudaMallocPitch ensure correct data structure alignment
  size_t fsmPitch, inputsPitch;   

  // Allocate the device memory
  cudaMallocPitch(&d_fsm, &fsmPitch, sizeof(int)*NUM_SYMBOLS, NUM_STATES);
  cudaMallocPitch(&d_inputs, &inputsPitch, sizeof(int)*CHUNK_SIZE, NUM_THREADS);  
  cudaMalloc(&d_threadResultStates, sizeof(int)*NUM_THREADS);

  // Bind Texture to GPU Cache
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
  cudaBindTexture2D(0, textureFSM, d_fsm, channelDesc, NUM_SYMBOLS, NUM_STATES, fsmPitch);

  // Copy FSM to GPU
  cudaMemcpy2D(d_fsm, fsmPitch, fsm, (NUM_SYMBOLS * sizeof(int)), (NUM_SYMBOLS * sizeof(int)), NUM_STATES, cudaMemcpyHostToDevice);

  while (inputStringIndex < inputLength) {

    inputIndex = 0;
    threadIndex = 0;
    loadingChunks = true;
    emptyChunk = false;

    // Chunkifying the input string 
    while (loadingChunks && threadIndex < NUM_THREADS) {
      chunkStartIndex = inputStringIndex;

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
          inputs[inputIndex][k] = inputString[chunkStartIndex + k];
        }

        inputIndex += 1;
        threadIndex += 1;
      } else {
        // Copy inputString chunk for every possible start state
        for (int j = 0; j < (inputStringIndex - chunkStartIndex); j++) {
          inputs[inputIndex][j] = inputString[chunkStartIndex + j];
        }

        inputIndex += 1;
        threadIndex += NUM_STATES;

        if (threadIndex >= NUM_THREADS) {
          threadIndex = NUM_THREADS;
          loadingChunks = false;
        }
      }
    }

    // Copy host memory to device memory
    cudaMemcpy2D(d_inputs, inputsPitch, *inputs, (CHUNK_SIZE * sizeof(int)), (CHUNK_SIZE * sizeof(int)), NUM_INPUTS, cudaMemcpyHostToDevice);

    // Run FSM on GPU
    gettimeofday(&kernelStart, NULL);
    EvaluateFSM<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_fsm, d_threadResultStates, d_inputs, fsmPitch, inputsPitch, NUM_STATES, NUM_SYMBOLS, CHUNK_SIZE, threadIndex, currentState);
    
    cudaThreadSynchronize();

    gettimeofday(&kernelEnd, NULL);
    kernelTime += ((kernelEnd.tv_sec - kernelStart.tv_sec) * 1000000u + kernelEnd.tv_usec - kernelStart.tv_usec) / 1.e6;
    // cudaPrintfDisplay(stdout, true); // Printf capabilities

    // Copy the data back to the host memory  
    cudaMemcpy(h_threadResultStates, d_threadResultStates, (NUM_THREADS * sizeof(int)), cudaMemcpyDeviceToHost);  

    // Find the new current state by using the enumerated thread results 
    // to see if we were able to find a valid future state
    currentState = h_threadResultStates[0];
    predictionLevel = 0;
    while ((1 + (predictionLevel*NUM_STATES) + currentState) < ( sizeof(h_threadResultStates) / sizeof(int) )) {
      if ((1 + (predictionLevel*NUM_STATES) + currentState) < threadIndex) {
        currentState = h_threadResultStates[1 + (predictionLevel*NUM_STATES) + currentState];
        predictionLevel += 1;
      } else {
        break;
      }
    }
  }

  cudaFree(d_fsm);

  cout << "GPU Final State: " << currentState << endl;
  cout << "Kernel Time: " << kernelTime << endl;

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