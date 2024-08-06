package org.yah.tools.gemm;

public interface Gemm extends AutoCloseable {

    default String name() {
        return getClass().getSimpleName();
    }

    Times times();

    @Override
    default void close() {
    }

    interface Sgemm extends Gemm {
        void sgemm(int M, int N, int K,
                   float alpha, float[] A, int lda,
                   float[] B, int ldb,
                   float beta, float[] C, int ldc);
    }

    interface Dgemm extends Gemm {
        void dgemm(int M, int N, int K,
                   double alpha, double[] A, int lda,
                   double[] B, int ldb,
                   double beta, double[] C, int ldc);

    }

}
