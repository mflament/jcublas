#define IDX2C(row,col,ld) ((col) * (ld) + (row))


template <typename T>
__device__ void gemm(int M, int N, int K,
                     const T alpha, const T *A, int lda,
                     const T *B, int ldb,
                     const T beta,
                     T *C, int ldc)
{
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += gridDim.x*blockDim.x)
    {
        for (int m = blockIdx.y * blockDim.y + threadIdx.y; m < M; m += gridDim.y*blockDim.y)
        {
            T c = 0;
            for (int k = 0; k < K; ++k)
            {
                c += A[IDX2C(m,k,lda)] * B[IDX2C(k,n,ldb)];
            }
            c = alpha * c;
            if (beta != 0)
                c = c + beta * C[IDX2C(m,n,ldc)];
            C[IDX2C(m,n,ldc)] = c;
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