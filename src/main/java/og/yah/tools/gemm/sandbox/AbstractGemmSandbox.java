package og.yah.tools.gemm.sandbox;

import og.yah.tools.cuda.cublas.CublasAPI;
import og.yah.tools.cuda.runtime.RuntimeAPI;
import og.yah.tools.gemm.CublasGemm.CublasSgemm;
import og.yah.tools.gemm.MatrixExecutor;

import java.util.Random;

import static og.yah.tools.gemm.CublasGemm.CublasDgemm;
import static og.yah.tools.gemm.JavaGemm.*;

public abstract class AbstractGemmSandbox {

    protected static RuntimeAPI cuda;
    protected static CublasAPI cublas;
    protected static MatrixExecutor matrixExecutor;

    protected AbstractGemmSandbox() {
    }

    protected final void execute() {
        setup();
        try {
            run();
        } finally {
            tearDown();
        }
    }

    protected void setup() {
        cuda = RuntimeAPI.load();
        cublas = CublasAPI.load();
        matrixExecutor = new MatrixExecutor();
    }

    protected void tearDown() {
        if (matrixExecutor != null) matrixExecutor.close();
    }

    protected abstract void run();

    protected static MatrixExecutor.Handler randomizer(long seed, float[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[IDX2C(row, col, ld)] = random.nextFloat();
    }

    protected static MatrixExecutor.Handler randomizer(long seed, double[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[IDX2C(row, col, ld)] = random.nextDouble();
    }

    public static abstract class AbstractSgemmSandbox extends AbstractGemmSandbox {
        protected static Sgemm singleThreadGemm;
        protected static Sgemm parallelizedGemm;
        protected static Sgemm parallelizedTransposedSgemm;
        protected static CublasSgemm cublasGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new SingleThreadSgemm();
            parallelizedGemm = new ParallelizedSgemm(matrixExecutor);
            parallelizedTransposedSgemm = new ParallelizedTransposedSgemm(matrixExecutor);
            cublasGemm = new CublasSgemm(cuda, cublas);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
        }
    }

    public static abstract class AbstractDgemmSandbox extends AbstractGemmSandbox {
        protected static Dgemm singleThreadGemm;
        protected static Dgemm parallelizedGemm;
        protected static Dgemm parallelizedTransposedSgemm;
        protected static CublasDgemm cublasGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new SingleThreadDgemm();
            parallelizedGemm = new ParallelizedDgemm(matrixExecutor);
            parallelizedTransposedSgemm = new ParallelizedTransposedDgemm(matrixExecutor);
            cublasGemm = new CublasDgemm(cuda, cublas);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
        }
    }
}
