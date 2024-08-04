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
import static og.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractSgemmSandbox;

@SuppressWarnings("unused")
public class SgemmTestSandbox extends AbstractSgemmSandbox {

    private static final double SGEMM_MAX_ERROR = 1E-5;
    private static final double DGEMM_MAX_ERROR = 1E-10;

    private TestSandboxSupport testSandboxSupport;

    public static void main(String[] args) {
        new SgemmTestSandbox().execute();
    }

    @Override
    protected void setup() {
        super.setup();
        testSandboxSupport = new TestSandboxSupport(matrixExecutor);
    }

    @Override
    protected void run() {
        testSgemm(500, 200, 400, 1, 0);
    }

    @SuppressWarnings("SameParameterValue")
    private void testSgemm(int M, int N, int K, float alpha, float beta) {
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] C = new float[M * N];
        double error;
        long cBaseSeed = 91011;

        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));

        float[] refC = new float[M * N];
        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, refC, M));
        singleThreadGemm.sgemm(M, N, K, alpha, A, M, B, K, beta, refC, M);

        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
        parallelizedGemm.sgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        error = testSandboxSupport.maxError(M, N, refC, C);
        if (error > SGEMM_MAX_ERROR)
            System.err.printf("%s error %s%n", parallelizedGemm, error);

        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
        parallelizedTransposedSgemm.sgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        error = testSandboxSupport.maxError(M, N, refC, C);
        if (error > SGEMM_MAX_ERROR)
            System.err.printf("%s error %s%n", parallelizedTransposedSgemm, error);

        matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
        cublasGemm.setConfig(null);
        cublasGemm.sgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
        error = testSandboxSupport.maxError(M, N, refC, C);
        if (error > SGEMM_MAX_ERROR)
            System.err.printf("%s error %.13f%n", cublasGemm, error);
        else if (error == 0)
            System.out.printf("Perfect match %s%n", cublasGemm);

        List<Integer> unsupportedAlgos = new ArrayList<>();
        for (int i = 0; i < 24; i++) {
            cublasGemmAlgo_t algo = NativeEnum.resolve(cublasGemmAlgo_t.class, i);
            cublasGemm.setConfig(new GemmConfig(cublasAtomicsMode_t.CUBLAS_ATOMICS_NOT_ALLOWED, NativeEnum.flags(cublasMath_t.CUBLAS_PEDANTIC_MATH, cublasMath_t.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),
                    cublasComputeType_t.CUBLAS_COMPUTE_32F_PEDANTIC, algo));
            matrixExecutor.parallelize(M, N, threadIndex -> randomizer(cBaseSeed + threadIndex, C, M));
            try {
                cublasGemm.sgemm(M, N, K, alpha, A, M, B, K, beta, C, M);
            } catch (CublasException e) {
                if (e.status() == cublasStatus_t.CUBLAS_STATUS_NOT_SUPPORTED) {
                    unsupportedAlgos.add(i);
                    continue;
                }
                throw e;
            }
            error = testSandboxSupport.maxError(M, N, refC, C);
            if (error > SGEMM_MAX_ERROR)
                System.err.printf("%s algo %d error %.13f%n", cublasGemm, i, error);
            else if (error == 0)
                System.out.printf("Perfect match %s algo %d%n", cublasGemm, i);
        }
        if (!unsupportedAlgos.isEmpty())
            System.out.printf("%s algo %s not supported%n", cublasGemm, unsupportedAlgos);
    }

}
