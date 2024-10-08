package org.yah.tools.gemm;

import org.yah.tools.gemm.Times.Operation;

import java.util.Objects;

public abstract class JavaGemm implements Gemm {

    @Override
    public String toString() {
        return name();
    }

    public static final class SingleThreadSgemm extends JavaGemm implements Sgemm {
        private final Times times;

        public SingleThreadSgemm() {
            this.times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public String name() {
            return "CPU(1T)";
        }

        @Override
        public GemmId id() {
            return GemmId.ST;
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            times.measure(Operation.GEMM, () -> {
                for (int n = 0; n < N; n++) {
                    for (int m = 0; m < M; m++) {
                        float temp = 0;
                        for (int k = 0; k < K; k++) {
                            temp += A[IDX2C(m, k, lda)] * B[IDX2C(k, n, ldb)];
                        }
                        temp = alpha * temp;
                        if (beta != 0)
                            temp = temp + beta * C[IDX2C(m, n, ldc)];
                        C[IDX2C(m, n, ldc)] = temp;
                    }
                }
            });
        }
    }

    public static final class SingleThreadDgemm extends JavaGemm implements Dgemm {
        private final Times times;

        public SingleThreadDgemm() {
            this.times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public GemmId id() {
            return GemmId.ST;
        }

        @Override
        public String name() {
            return "CPU(1T)";
        }

        @Override
        public void dgemm(int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc) {
            times.measure(Operation.GEMM, () -> {
                for (int n = 0; n < N; n++) {
                    for (int m = 0; m < M; m++) {
                        double temp = 0;
                        for (int k = 0; k < K; k++) {
                            temp += A[IDX2C(m, k, lda)] * B[IDX2C(k, n, ldb)];
                        }
                        temp = temp * alpha;
                        if (beta != 0)
                            temp = temp + beta * C[IDX2C(m, n, ldc)];
                        C[IDX2C(m, n, ldc)] = temp;
                    }
                }
            });
        }
    }

    public static final class ParallelizedSgemm extends JavaGemm implements Sgemm {
        private final MatrixExecutor executor;

        private final Times times;

        public ParallelizedSgemm(MatrixExecutor executor) {
            this.executor = Objects.requireNonNull(executor, "executor is null");
            this.times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public GemmId id() {
            return GemmId.MT;
        }

        @Override
        public String name() {
            return String.format("CPU(%dT)", executor.threads());
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            times.measure(Operation.GEMM, () ->
                    executor.parallelize(M, N, (row, col) -> cellGemm(K, alpha, A, lda, B, ldb, beta, C, ldc, row, col))
            );
        }

        private static void cellGemm(int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc,
                                     int row, int col) {
            float value = 0;
            for (int k = 0; k < K; ++k) {
                value += A[IDX2C(row, k, lda)] * B[IDX2C(k, col, ldb)];
            }
            value = alpha * value;
            if (beta != 0)
                value = value + beta * C[IDX2C(row, col, ldc)];
            C[IDX2C(row, col, ldc)] = value;
        }

    }

    public static final class ParallelizedDgemm extends JavaGemm implements Dgemm {
        private final MatrixExecutor executor;
        private final Times times;

        public ParallelizedDgemm(MatrixExecutor executor) {
            this.executor = Objects.requireNonNull(executor, "executor is null");
            times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public GemmId id() {
            return GemmId.MT;
        }

        @Override
        public String name() {
            return String.format("CPU(%dT)", executor.threads());
        }

        @Override
        public void dgemm(int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc) {
            times.measure(Operation.GEMM, () ->
                    executor.parallelize(M, N, (row, col) -> cellGemm(K, alpha, A, lda, B, ldb, beta, C, ldc, row, col))
            );
        }

        private static void cellGemm(int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc,
                                     int row, int col) {
            double value = 0;
            for (int k = 0; k < K; ++k) {
                value += A[IDX2C(row, k, lda)] * B[IDX2C(k, col, ldb)];
            }
            value = alpha * value;
            if (beta != 0)
                value = value + beta * C[IDX2C(row, col, ldc)];
            C[IDX2C(row, col, ldc)] = value;
        }
    }

    public static final class ParallelizedTransposedSgemm extends JavaGemm implements Sgemm {
        private final MatrixExecutor executor;
        private float[] transposedA;
        private final Times times;

        public ParallelizedTransposedSgemm(MatrixExecutor executor) {
            this.executor = Objects.requireNonNull(executor, "executor is null");
            times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public GemmId id() {
            return GemmId.MT_TRANSPOSED;
        }

        @Override
        public String name() {
            return String.format("CPU(%dT)+TR", executor.threads());
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            times.measure(Operation.TRANSPOSE, () -> {
                if (transposedA == null || transposedA.length < A.length)
                    transposedA = new float[A.length];
                JavaGemm.transpose(executor, M, K, transposedA, A, lda);
            });
            times.measure(Operation.GEMM, () ->
                    executor.parallelize(M, N, (row, col) -> cellGemm(K, alpha, transposedA, K, B, ldb, beta, C, ldc, row, col))
            );
        }

        private static void cellGemm(int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc,
                                     int row, int col) {
            float value = 0;
            for (int k = 0; k < K; ++k) {
                value += A[IDX2R(row, k, lda)] * B[IDX2C(k, col, ldb)];
            }
            value = alpha * value;
            if (beta != 0)
                value = value + beta * C[IDX2C(row, col, ldc)];
            C[IDX2C(row, col, ldc)] = value;
        }

    }

    public static final class ParallelizedTransposedDgemm extends JavaGemm implements Dgemm {
        private final MatrixExecutor executor;
        private double[] transposedA;
        private final Times times;

        public ParallelizedTransposedDgemm(MatrixExecutor executor) {
            this.executor = Objects.requireNonNull(executor, "executor is null");
            times = new Times(name());
        }

        @Override
        public Times times() {
            return times;
        }

        @Override
        public GemmId id() {
            return GemmId.MT_TRANSPOSED;
        }

        @Override
        public String name() {
            return String.format("CPU(%dT)+TR", executor.threads());
        }

        @Override
        public void dgemm(int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc) {
            times.measure(Operation.TRANSPOSE, () -> {
                if (transposedA == null || transposedA.length < A.length)
                    transposedA = new double[A.length];
                JavaGemm.transpose(executor, M, K, transposedA, A, lda);
            });
            times.measure(Operation.GEMM,
                    () -> executor.parallelize(M, N, (row, col) -> cellGemm(K, alpha, transposedA, K, B, ldb, beta, C, ldc, row, col)));
        }

        private static void cellGemm(int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc,
                                     int row, int col) {
            double value = 0;
            for (int k = 0; k < K; ++k) {
                value += A[IDX2R(row, k, lda)] * B[IDX2C(k, col, ldb)];
            }
            value = alpha * value;
            if (beta != 0)
                value = value + beta * C[IDX2C(row, col, ldc)];
            C[IDX2C(row, col, ldc)] = value;
        }
    }

    public static void transpose(MatrixExecutor matrixExecutor, int rows, int cols,
                                 float[] dst, float[] src, int ld) {
        matrixExecutor.parallelize(rows, cols, (row, col) -> dst[IDX2C(col, row, cols)] = src[IDX2C(row, col, ld)]);
    }

    public static void transpose(MatrixExecutor matrixExecutor, int rows, int cols,
                                 double[] dst, double[] src, int ld) {
        matrixExecutor.parallelize(rows, cols, (row, col) -> dst[IDX2C(col, row, cols)] = src[IDX2C(row, col, ld)]);
    }

    public static int IDX2C(int row, int col, int ld) {
        return col * ld + row;
    }

    public static int IDX2R(int row, int col, int ld) {
        return row * ld + col;
    }
}
