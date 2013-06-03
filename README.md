shmalloc
========

Dynamic `__shared__` memory allocation for CUDA

CUDA currently provides two avenues for allocating `__shared__` memory: static allocation via `__shared__` arrays and a single dynamically-allocated block which must sized at kernel launch time.

These two methods are often suitable for simple kernels, but as codebases grow they do not scale well.

For example, as kernels grow in sophistication, it becomes difficult to predict in advance `__shared__` memory requirements at the kernel launch site:

    some_complicated_kernel<<<num_blocks, block_size, what goes here??>>>(...);

Working around this problem requires the kernel implementation and kernel launch site to engage in an ad hoc allocation negotiation which is at best difficult to maintain and at worst impossible.

Statically allocated arrays also introduce problems for libraries of generic code. Take for example a generic `sum` library function:

    template<unsigned int BLOCK_SIZE, typename T>
    __device__ T sum(T *ptr, int n)
    {
      __shared__ T s_data[BLOCK_SIZE];
    
      // reduce partial sums per thread
      ...
      __syncthreads();
    
      // perform reduction tree
      ...
      __syncthreads();
    
      return s_data[0];
    } // end reduce()

In order to allocate the `__shared__` array `s_data`, `sum` requires its caller
to invoke it using the CUDA block size as a template parameter. If the kernel's
block size is dynamic, this is impossible. Because tuning for different
hardware architectures may require different CUDA block sizes, block sizes
are dynamic in general.

If the caller does have compile-time knowlege of the CUDA block size, then it
can invoke `sum` using the block size as a parameter. However, if this size
causes the `s_data` array to overflow `__shared__` memory, compilation will
fail.

We could rewrite `sum`'s interface to require an extra buffer:

    template<typename T>
    __device__ T sum(T *ptr, int n, T *buffer);

However, this leaks implementation details and requires an additional protocol for communicating resource requirements to the caller. Nothing is actually solved; we have simply pushed the allocation problem onto the caller.

`shmalloc` solves these problems by virtualizing `__shared__` memory and
providing a robust means of dynamically managing it: if insufficient on-chip
`__shared__` memory resources are available, we proceed with degraded
performance by procuring resources from a backing store; in this case, global
memory.

In this way, virtualization of `__shared__` memory transforms the dynamic on-chip memory allocation kernel launch parameter from a hard functional requirement into a softer performance knob.

The following snippet demonstrates how to use `shmalloc` and `shfree` to manage dynamically-allocated `__shared__` memory in a reduction kernel:

    #include "shmalloc.hpp"

    __global__ void sum_kernel(int *data, int *result, size_t max_data_segment_size)
    {
      // thread 0 intializes the __shared__ memory heap with the dynamic size of on-chip memory
      if(threadIdx.x == 0)
      {
        init_on_chip_malloc(max_data_segment_size);
      }
      __syncthreads();
    
      __shared__ int *s_s_data;
    
      unsigned int n = blockDim.x;
    
      // shmalloc isn't thread-safe, so make sure only thread 0 calls it
      // note that the pointer is communicated through the statically allocated
      // __shared__ variable s_s_data
      if(threadIdx.x == 0)
      {
        s_s_data = static_cast<int*>(shmalloc(n * sizeof(int)));
      }
      __syncthreads();
    
      int *s_data = s_s_data;
    
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

