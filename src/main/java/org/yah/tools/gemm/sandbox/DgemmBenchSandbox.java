package org.yah.tools.gemm.sandbox;

import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractDgemmSandbox;

public class DgemmBenchSandbox extends AbstractDgemmSandbox {

    private static final int runs = 3;

    private double[] A;
    private double[] B;
    private double[] C;

    public static void main(String[] args) {
        new DgemmBenchSandbox().execute(args);
    }

    @Override
    protected void setup() {
        super.setup();
        A = new double[M * K];
        B = new double[K * N];
        C = new double[M * N];

        System.out.println("generate inputs");
        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));
    }

    protected void run() {
//        bench(singleThreadGemm);
        bench(parallelizedGemm);
        bench(parallelizedTransposedSgemm);
        bench(cublasGemm);
        bench(cudaGemm);
        bench(cudaTiledGemm);
    }

    private void bench(Gemm.Dgemm gemm) {
        System.out.print(gemm);
        for (int i = 0; i < runs; i++) {
            gemm.dgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.println(" " + gemm.times());
    }

}
