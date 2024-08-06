package org.yah.tools.gemm.sandbox;

import org.yah.tools.cuda.cublas.CublasException;
import org.yah.tools.cuda.cublas.cublasAtomicsMode_t;
import org.yah.tools.cuda.cublas.cublasComputeType_t;
import org.yah.tools.cuda.cublas.cublasGemmAlgo_t;
import org.yah.tools.cuda.cublas.cublasMath_t;
import org.yah.tools.cuda.cublas.cublasStatus_t;
import org.yah.tools.cuda.jna.NativeEnum;
import org.yah.tools.gemm.CublasGemm;

import static org.yah.tools.gemm.CublasGemm.GemmConfig;
import static org.yah.tools.gemm.Gemm.Dgemm;
import static org.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractDgemmSandbox;

@SuppressWarnings("unused")
public class DgemmTestSandbox extends AbstractDgemmSandbox {

    private static final double MAX_ERROR = 1E-10;

    private static final long C_BASE_SEED = 91011;

    private TestSandboxSupport testSandboxSupport;

    private double[] A;
    private double[] B;
    private double[] C;
    private double[] Cref;

    public static void main(String[] args) {
        new DgemmTestSandbox().execute(args);
    }

    @Override
    protected void setup() {
        super.setup();
        testSandboxSupport = new TestSandboxSupport(matrixExecutor);
        A = new double[M * K];
        B = new double[K * N];
        C = new double[M * N];
        Cref = new double[M * N];

        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));
    }

    @Override
    protected void run() {
        testGemm(1, 0);
        testGemm(5, 0);
        testGemm(1, 10);
        System.out.println("Success");
    }

    @SuppressWarnings("SameParameterValue")
    private void testGemm(double alpha, double beta) {
        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(C_BASE_SEED + threadIndex, Cref, M));
        singleThreadGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, Cref, M);

        testGemm(alpha, beta, parallelizedGemm);

        testGemm(alpha, beta, parallelizedTransposedSgemm);

        cublasGemm.setConfig(null);
        testGemm(alpha, beta, cublasGemm);
        for (int i = 0; i < 24; i++) {
            testGemm(alpha, beta, cublasGemm, i);
        }

        testGemm(alpha, beta, cudaGemm);
        testGemm(alpha, beta, cudaTiledGemm);
        testGemm(alpha, beta, cudaTransposedGemm);
    }

    private void testGemm(double alpha, double beta, Dgemm gemm) {
        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(C_BASE_SEED + threadIndex, C, M));
        gemm.dgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        double error = testSandboxSupport.maxError(M, N, Cref, C);
        if (error > MAX_ERROR)
            throw new AssertionError(gemm + " error : " + error);
    }

    private void testGemm(double alpha, double beta, CublasGemm.CublasDgemm gemm, int algo) {
        cublasGemm.setConfig(new GemmConfig(cublasAtomicsMode_t.CUBLAS_ATOMICS_NOT_ALLOWED,
                NativeEnum.flags(cublasMath_t.CUBLAS_PEDANTIC_MATH, cublasMath_t.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),
                cublasComputeType_t.CUBLAS_COMPUTE_32F_PEDANTIC, NativeEnum.resolve(cublasGemmAlgo_t.class, algo)));
        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(C_BASE_SEED + threadIndex, C, M));
        try {
            cublasGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        } catch (CublasException e) {
            if (e.status() == cublasStatus_t.CUBLAS_STATUS_NOT_SUPPORTED)
                return;
            throw e;
        }
        double error = testSandboxSupport.maxError(M, N, Cref, C);
        if (error > MAX_ERROR)
            throw new AssertionError(gemm + "(algo " + algo + ") error : " + error);
    }

}
