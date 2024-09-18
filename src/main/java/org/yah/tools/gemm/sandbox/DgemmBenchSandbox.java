package org.yah.tools.gemm.sandbox;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractDgemmSandbox;

public class DgemmBenchSandbox extends AbstractDgemmSandbox {

    private static final Logger LOGGER = LoggerFactory.getLogger(DgemmBenchSandbox.class);

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

        long start = System.nanoTime();
        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));
        LOGGER.info("generate inputs: {}ms", formatTime((System.nanoTime() - start) * 1E-6, 1));
    }

    protected void run() {
        bench(singleThreadGemm);
        bench(parallelizedGemm);
        bench(parallelizedTransposedSgemm);
        bench(cublasGemm);
        bench(cudaGemm);
        bench(cudaTiledGemm);
        bench(cudaTransposedGemm);
        System.out.println(formatTimes());
        exportCSV("dgemm.csv");
    }

    private void bench(Gemm.Dgemm gemm) {
        bench(gemm, g -> g.dgemm(M, N, K, 1, A, M, B, K, 0, C, M));
    }

}
