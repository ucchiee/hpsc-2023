#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
using namespace cooperative_groups;

// #define DEBUG 1

__global__ void scan(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();
  for(int j=1; j<N; j<<=1) {
    b[i] = a[i];
    grid.sync();
    a[i] += b[i-j];
    grid.sync();
  }
}

__global__ void init_bucket(int *bucket, int range) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= range) return;
  bucket[i] = 0;
}

__global__ void count_key(int *key, int *bucket, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= n) return;
  atomicAdd(bucket + key[i], 1);
}

__global__ void write_key(int *key, int start, int end, int val, int n) {
  // bucket[start:end] = val;
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= n) return;
  if (start <= i && i < end) key[i] = val;
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket, *buf;
  cudaMallocManaged(&bucket, range*sizeof(int));  
  cudaMallocManaged(&buf, range*sizeof(int));  

  int m = 1024;
  init_bucket<<<(range + m - 1)/m, m>>>(bucket, range);
  cudaDeviceSynchronize();

  count_key<<<(n + m - 1)/m, m>>>(key, bucket, n);
  cudaDeviceSynchronize();

#ifdef DEBUG
  // bucket before scan
  for (int i=0; i<range; i++) {
    printf("%d ", bucket[i]);
  }
  printf("\n");
#endif

  // bucket を scan する
  void *args[] = {(void *)&bucket, (void *)&buf, (void *)&n};
  cudaLaunchCooperativeKernel((void*)scan, (n + m - 1)/m, m, args);
  cudaDeviceSynchronize();

#ifdef DEBUG
  // bucket after scan
  for (int i=0; i<range; i++) {
    printf("%d ", bucket[i]);
  }
  printf("\n");
#endif

  for (int val=0; val<range; val++) {
    if (val == 0) {
      write_key<<<(n + m - 1)/m, m>>>(key, 0, bucket[val], val, n);
    } else {
      write_key<<<(n + m - 1)/m, m>>>(key, bucket[val-1], bucket[val], val, n);
    }

  }
  // それぞれ異なる場所に書き込むため、毎回同期を取る必要はない
  // write_key() 内で同期を取る必要もない
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(bucket);
  cudaFree(key);
}
