package og.yah.tools.gemm.sandbox;

import og.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractDgemmSandbox;

public class DgemmBenchSandbox extends AbstractDgemmSandbox {

    private static final int M = 2000;
    private static final int N = 1000;
    private static final int K = 1500;

    private static final int runs = 5;

    public static void main(String[] args) {
        new DgemmBenchSandbox().execute();
    }

    protected void run() {
        double[] A = new double[M * K];
        double[] B = new double[K * N];
        double[] C = new double[M * N];

        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));

        for (int i = 0; i < runs; i++) {
            singleThreadGemm.dgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", singleThreadGemm, singleThreadGemm.times());

        for (int i = 0; i < runs; i++) {
            parallelizedGemm.dgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", parallelizedGemm, parallelizedGemm.times());

        for (int i = 0; i < runs; i++) {
            parallelizedTransposedSgemm.dgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", parallelizedTransposedSgemm, parallelizedTransposedSgemm.times());

        for (int i = 0; i < runs; i++) {
            cublasGemm.dgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.printf("%s : %s%n", cublasGemm, cublasGemm.times());
    }

}
