package org.yah.tools.gemm.sandbox;

import org.yah.tools.cuda.cublas.CublasAPI;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.gemm.CublasGemm.CublasSgemm;
import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.JavaGemm;
import org.yah.tools.gemm.MatrixExecutor;

import java.util.Random;

import static org.yah.tools.gemm.CublasGemm.CublasDgemm;
import static org.yah.tools.gemm.CudaGemm.CudaDgemm;
import static org.yah.tools.gemm.CudaGemm.CudaSgemm;

public abstract class AbstractGemmSandbox {

    public static final int MAX_ARRAY_LENGTH = Integer.MAX_VALUE - 2;
    protected static RuntimeAPI cuda;
    protected static CublasAPI cublas;
    protected static MatrixExecutor matrixExecutor;

    private static final double MB = 1024 * 1024;

    protected int cudaDevice = 0;
    protected int M = 2000;
    protected int N = 1000;
    protected int K = 1500;

    protected AbstractGemmSandbox() {
    }

    protected final void execute(String[] args) {
        parseCommandLine(args);
        setup();
        try {
            run();
        } finally {
            tearDown();
        }
    }

    protected void parseCommandLine(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-M" -> M = Integer.parseInt(args[++i]);
                case "-N" -> N = Integer.parseInt(args[++i]);
                case "-K" -> K = Integer.parseInt(args[++i]);
                default -> {
                    System.err.println("Invalid parameter " + args[i]);
                    System.exit(1);
                }
            }
        }
    }

    protected void setup() {
        validateDims();
        System.out.printf("A=%s B=%s C=%s%n", matrixInfo(M, K), matrixInfo(K, N), matrixInfo(M, N));
        cuda = RuntimeAPI.load();
        cublas = CublasAPI.load();
        matrixExecutor = new MatrixExecutor();
        cuda.cudaSetDevice(cudaDevice);
    }

    private String matrixInfo(int rows, int cols) {
        return String.format("%dx%d (%.2fMB)", rows, cols, rows * cols * elementSize() / MB);
    }

    private void validateDims() {
        validateDim("A", M * K);
        validateDim("B", K * N);
        validateDim("C", M * N);
    }

    private void validateDim(String matrix, int dim) {
        if (dim > +MAX_ARRAY_LENGTH) // max array dim
            throw new IllegalStateException(matrix + " array length " + dim + " overflow max length " + MAX_ARRAY_LENGTH);
    }

    protected void tearDown() {
        if (matrixExecutor != null) matrixExecutor.close();
    }

    protected abstract void run();

    protected static MatrixExecutor.Handler randomizer(long seed, float[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[JavaGemm.IDX2C(row, col, ld)] = random.nextFloat();
    }

    protected static MatrixExecutor.Handler randomizer(long seed, double[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[JavaGemm.IDX2C(row, col, ld)] = random.nextDouble();
    }

    protected final String formatSizes(int M, int N, int K) {
        return String.format("A=%.2fMB B=%.2fMB C=%.2fMB",
                M * K * elementSize() / MB,
                K * N * elementSize() / MB,
                M * N * elementSize() / MB);
    }

    protected abstract long elementSize();

    public static abstract class AbstractSgemmSandbox extends AbstractGemmSandbox {
        protected static Gemm.Sgemm singleThreadGemm;
        protected static Gemm.Sgemm parallelizedGemm;
        protected static Gemm.Sgemm parallelizedTransposedSgemm;
        protected static CublasSgemm cublasGemm;
        protected static CudaSgemm cudaGemm;
        protected static CudaSgemm cudaTiledGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new JavaGemm.SingleThreadSgemm();
            parallelizedGemm = new JavaGemm.ParallelizedSgemm(matrixExecutor);
            parallelizedTransposedSgemm = new JavaGemm.ParallelizedTransposedSgemm(matrixExecutor);
            cublasGemm = new CublasSgemm(cuda, cublas);
            cudaGemm = new CudaSgemm(cuda, cudaDevice, false);
            cudaTiledGemm = new CudaSgemm(cuda, cudaDevice, true);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
            if (cudaGemm != null) cudaGemm.close();
        }

        @Override
        protected final long elementSize() {
            return Float.BYTES;
        }
    }

    public static abstract class AbstractDgemmSandbox extends AbstractGemmSandbox {
        protected static Gemm.Dgemm singleThreadGemm;
        protected static Gemm.Dgemm parallelizedGemm;
        protected static Gemm.Dgemm parallelizedTransposedSgemm;
        protected static CublasDgemm cublasGemm;
        protected static CudaDgemm cudaGemm;
        protected static CudaDgemm cudaTiledGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new JavaGemm.SingleThreadDgemm();
            parallelizedGemm = new JavaGemm.ParallelizedDgemm(matrixExecutor);
            parallelizedTransposedSgemm = new JavaGemm.ParallelizedTransposedDgemm(matrixExecutor);
            cublasGemm = new CublasDgemm(cuda, cublas);
            cudaGemm = new CudaDgemm(cuda, cudaDevice, false);
            cudaTiledGemm = new CudaDgemm(cuda, cudaDevice, true);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
            if (cudaGemm != null) cudaGemm.close();
        }

        @Override
        protected final long elementSize() {
            return Double.BYTES;
        }
    }
}
