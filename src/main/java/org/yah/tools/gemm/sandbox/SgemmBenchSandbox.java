package org.yah.tools.gemm.sandbox;

import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractSgemmSandbox;

public class SgemmBenchSandbox extends AbstractSgemmSandbox {

    private static final int runs = 3;

    private float[] A;
    private float[] B;
    private float[] C;

    public static void main(String[] args) {
        new SgemmBenchSandbox().execute(args);
    }

    @Override
    protected void setup() {
        super.setup();
        A = new float[M * K];
        B = new float[K * N];
        C = new float[M * N];

        System.out.println("generate inputs");
        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));
    }

    @Override
    protected void run() {
//        bench(singleThreadGemm);
        bench(parallelizedGemm);
        bench(parallelizedTransposedSgemm);
        bench(cublasGemm);
        bench(cudaGemm);
        bench(cudaTiledGemm);
    }

    private void bench(Gemm.Sgemm gemm) {
        System.out.print(gemm);
        for (int i = 0; i < runs; i++) {
            gemm.sgemm(M, N, K, 1, A, M, B, K, 0, C, M);
        }
        System.out.println(" " + gemm.times());
    }

}
