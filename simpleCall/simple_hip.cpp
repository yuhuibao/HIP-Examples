/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define NUM 1048576

#define THREADS_PER_BLOCK_X 16

__device__ __declspec(noinline) int plusone(int pcounter) {
  pcounter = pcounter + 1;
  return pcounter;
}

__global__ __declspec(noinline) void simple(float *__restrict__ a)

{
  int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  a[tid] = plusone(tid);
}

using namespace std;

int main() {

  float *hostA;
  float *deviceA;
  int grid_size = (NUM - 1) / THREADS_PER_BLOCK_X + 1;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl;

  int i;
  int errors;

  hostA = (float *)malloc(NUM * sizeof(float));

  HIP_ASSERT(hipMalloc((void **)&deviceA, NUM * sizeof(float)));

  hipLaunchKernelGGL(simple, grid_size, THREADS_PER_BLOCK_X, 0, 0,
                     deviceA);

  HIP_ASSERT(
      hipMemcpy(hostA, deviceA, NUM * sizeof(float), hipMemcpyDeviceToHost));

  // verify the results

  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != i + 1) {
      errors++;
    }
  }
  if (errors != 0) {
    printf("FAILED: %d errors\n", errors);
  } else {
    printf("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  // hipResetDefaultAccelerator();

  return errors;
}
