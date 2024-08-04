package og.yah.tools.gemm.sandbox;

import og.yah.tools.cuda.cublas.CublasException;
import og.yah.tools.cuda.cublas.cublasAtomicsMode_t;
import og.yah.tools.cuda.cublas.cublasComputeType_t;
import og.yah.tools.cuda.cublas.cublasGemmAlgo_t;
import og.yah.tools.cuda.cublas.cublasMath_t;
import og.yah.tools.cuda.cublas.cublasStatus_t;
import og.yah.tools.cuda.jna.NativeEnum;

import java.util.ArrayList;
import java.util.List;

import static og.yah.tools.gemm.CublasGemm.GemmConfig;
import static og.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractDgemmSandbox;

@SuppressWarnings("unused")
public class DgemmTestSandbox extends AbstractDgemmSandbox {

    private static final double SGEMM_MAX_ERROR = 1E-5;
    private static final double DGEMM_MAX_ERROR = 1E-10;

    private TestSandboxSupport testSandboxSupport;

    public static void main(String[] args) {
        new DgemmTestSandbox().execute();
    }

    @Override
    protected void setup() {
        super.setup();
        testSandboxSupport = new TestSandboxSupport(matrixExecutor);
    }

    @Override
    protected void run() {
        testDgemm(500, 200, 400, 1, 0);
    }

    @SuppressWarnings("SameParameterValue")
    private void testDgemm(int M, int N, int K, double alpha, double beta) {
        double[] A = new double[M * K];
        double[] B = new double[K * N];
        double[] C = new double[M * N];
        double error;
        long cBaseSeed = 91011;

        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));

        double[] refC = new double[M * N];
        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, refC, M));
        singleThreadGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, refC, M);

        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
        parallelizedGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        error = testSandboxSupport.maxError(M, N, refC, C);
        if (error > DGEMM_MAX_ERROR)
            System.err.printf("ParallelizedGemm error %s%n", error);

        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
        cublasGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        error = testSandboxSupport.maxError(M, N, refC, C);
        if (error > DGEMM_MAX_ERROR)
            System.err.printf("CublasGemm error %.13f%n", error);
        else if (error == 0)
            System.out.printf("Perfect match CublasGemm%n");

        List<Integer> unsupportedAlgos = new ArrayList<>();
        for (int i = 0; i < 24; i++) {
            cublasGemmAlgo_t algo = NativeEnum.resolve(cublasGemmAlgo_t.class, i);
            cublasGemm.setConfig(new GemmConfig(cublasAtomicsMode_t.CUBLAS_ATOMICS_NOT_ALLOWED,
                    NativeEnum.flags(cublasMath_t.CUBLAS_PEDANTIC_MATH, cublasMath_t.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),
                    cublasComputeType_t.CUBLAS_COMPUTE_64F_PEDANTIC, algo));
            matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
            try {
                cublasGemm.dgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
            } catch (CublasException e) {
                if (e.status() == cublasStatus_t.CUBLAS_STATUS_NOT_SUPPORTED) {
                    unsupportedAlgos.add(i);
                    continue;
                }
                throw e;
            }
            error = testSandboxSupport.maxError(M, N, refC, C);
            if (error > DGEMM_MAX_ERROR)
                System.err.printf("CublasGemm algo %d error %.13f%n", i, error);
            else if (error == 0)
                System.out.printf("Perfect match CublasGemm algo %d%n", i);
        }
        if (!unsupportedAlgos.isEmpty())
            System.out.printf("dgemm cublas algo %s not supported%n", unsupportedAlgos);
    }
}
