#include <cstdio>
#include <cassert>
#include <thrust/device_vector.h>
#include <iostream>
#include "shmalloc.hpp"


__global__ void sum_kernel(int *data, int *result, size_t max_data_segment_size)
{
  // thread 0 intializes the __shared__ memory heap with the dynamic size of on-chip memory
  if(threadIdx.x == 0)
  {
    init_on_chip_malloc(max_data_segment_size);
  }
  __syncthreads();

  __shared__ int *s_data;

  unsigned int n = blockDim.x;

  // shmalloc isn't thread-safe, so make sure only thread 0 calls it
  // note that the pointer is communicated through the statically allocated
  // __shared__ variable s_data
  if(threadIdx.x == 0)
  {
    s_data = static_cast<int*>(shmalloc(n * sizeof(int)));
  }
  __syncthreads();

  for(int i = threadIdx.x; i < n; i += blockDim.x)
  {
    s_data[i] = data[i];
  }

  __syncthreads();

  while(n > 1)
  {
    unsigned int half_n = n / 2;

    if(threadIdx.x < half_n)
    {
      s_data[threadIdx.x] += s_data[n - threadIdx.x - 1];
    }

    __syncthreads();

    n -= half_n;
  }

  __syncthreads();
  if(threadIdx.x == 0)
  {
    *result = s_data[0];

    shfree(s_data);
  }
}


int main()
{
  size_t n = 512;

  thrust::device_vector<int> vec(n, 1);

  thrust::device_vector<int> sum(1,1);

  size_t smem_size = n * sizeof(int);

  // allocate exactly as much on-chip shared memory as we will ask for
  // this won't be enough to keep everything on chip due to bookkeeping shmalloc needs to perform
  sum_kernel<<<1,n,smem_size>>>(thrust::raw_pointer_cast(vec.data()),
                                thrust::raw_pointer_cast(sum.data()),
                                smem_size);
  std::cerr << "CUDA error: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  assert(n == sum[0]);

  // allocate twice as much on-chip shared memory as we will ask for
  // this should be enough to keep everything on chip
  smem_size = 2 * n * sizeof(int);
  sum_kernel<<<1,n,smem_size>>>(thrust::raw_pointer_cast(vec.data()),
                                thrust::raw_pointer_cast(sum.data()),
                                smem_size);
  std::cerr << "CUDA error: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  assert(n == sum[0]);

  // allocate half as much on-chip shared memory as we will ask for
  // this won't be enough to keep everything on chip
  smem_size = n * sizeof(int) / 2;
  sum_kernel<<<1,n,smem_size>>>(thrust::raw_pointer_cast(vec.data()),
                                thrust::raw_pointer_cast(sum.data()),
                                smem_size);
  std::cerr << "CUDA error: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  assert(n == sum[0]);

  // allocate no on-chip smem at all
  smem_size = 0;
  sum_kernel<<<1,n,smem_size>>>(thrust::raw_pointer_cast(vec.data()),
                                thrust::raw_pointer_cast(sum.data()),
                                smem_size);
  std::cerr << "CUDA error: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  assert(n == sum[0]);

  std::cout << "OK" << std::endl;

  return 0;
}

