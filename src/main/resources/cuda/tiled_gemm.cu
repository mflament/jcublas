#define IDX2C(row,col,ld) ((col) * (ld) + (row))

#if not defined BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

template <typename T>
__device__ void gemm(int M, int N, int K,
                     const T alpha, const T *A, int lda,
                     const T *B, int ldb,
                     const T beta,
                     T *C, int ldc)
{
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x,y;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int n = blockIdx.x * BLOCK_SIZE; n < N; n += gridDim.x * BLOCK_SIZE)
    {
        for (int m = blockIdx.y * BLOCK_SIZE; m < M; m += gridDim.y * BLOCK_SIZE)
        {
            T Csub = 0;
            for (int k = 0; k < K; k += BLOCK_SIZE)
            {
                // Load the matrices from device memory
                // to shared memory; each thread loads
                // one element of each matrix
                x = k + tx;
                y = m + ty;
                As[ty][tx] = x < K && y < M ? A[IDX2C(y,x,lda)] : 0;

                x = n + tx;
                y = k + ty;
                Bs[ty][tx] = x < N && y < K ? B[IDX2C(y,x,ldb)] : 0;

                // Synchronize to make sure the matrices are loaded
                __syncthreads();

                // Multiply the two matrices together;
                // each thread computes one element
                // of the block sub-matrix
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; ++i) {
                    Csub += As[ty][i] * Bs[i][tx];
                }

                // Synchronize to make sure that the preceding
                // computation is done before loading two new
                // sub-matrices of A and B in the next iteration
                __syncthreads();
            }

            // Write the block sub-matrix to device memory;
            // each thread writes one element
            x = n+tx;
            y = m+ty;
            if (x < N && y < M)
            {
                Csub = alpha * Csub;
                if (beta != 0)
                    Csub = Csub + beta * C[IDX2C(y,x,ldc)];
                C[IDX2C(y,x,ldc)] = Csub;
            }
        }
    }
}

extern "C" {
    __global__ void sgemm(int M, int N, int K,
                          const float alpha, const float *A, int lda,
                          const float *B, int ldb,
                          const float beta,
                          float *C, int ldc)
    {
        gemm<float>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
    }

    __global__ void dgemm(int M, int N, int K, const double alpha,
                          const double *A, int lda,
                          const double *B, int ldb,
                          const double beta,
                          double *C, int ldc)
    {
        gemm<double>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
    }
}