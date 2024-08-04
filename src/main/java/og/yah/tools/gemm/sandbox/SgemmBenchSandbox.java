package og.yah.tools.gemm.sandbox;

import og.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractSgemmSandbox;

public class SgemmBenchSandbox extends AbstractSgemmSandbox {

    private static final int M = 2000;
    private static final int N = 1000;
    private static final int K = 1500;

    private static final int runs = 5;

    public static void main(String[] args) {
        new SgemmBenchSandbox().execute();
    }

    @Override
    protected void run() {
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] C = new float[M * N];

        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));

        for (int i = 0; i < runs; i++) {
            singleThreadGemm.sgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", singleThreadGemm, singleThreadGemm.times());

        for (int i = 0; i < runs; i++) {
            parallelizedGemm.sgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", parallelizedGemm, parallelizedGemm.times());

        for (int i = 0; i < runs; i++) {
            parallelizedTransposedSgemm.sgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", parallelizedTransposedSgemm, parallelizedTransposedSgemm.times());

        for (int i = 0; i < runs; i++) {
            cublasGemm.sgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", cublasGemm, cublasGemm.times());
    }

}
