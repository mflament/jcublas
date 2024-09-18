package org.yah.tools.gemm.sandbox;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.sandbox.AbstractGemmSandbox.AbstractSgemmSandbox;

public class SgemmBenchSandbox extends AbstractSgemmSandbox {

    private static final Logger LOGGER = LoggerFactory.getLogger(SgemmBenchSandbox.class);

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

        long start = System.nanoTime();
        matrixExecutor.parallelize(M, K, threadIndex -> randomizer(1234 + threadIndex, A, M));
        matrixExecutor.parallelize(K, N, threadIndex -> randomizer(5678 + threadIndex, B, K));
        LOGGER.info("generate inputs: {}ms", formatTime((System.nanoTime() - start) * 1E-6, 1));
    }

    @Override
    protected void run() {
        bench(singleThreadGemm);
        bench(parallelizedGemm);
        bench(parallelizedTransposedSgemm);
        bench(cublasGemm);
        bench(cudaGemm);
        bench(cudaTiledGemm);
        bench(cudaTransposedGemm);
        System.out.println(formatTimes());
        exportCSV("sgemm.csv");
    }

    private void bench(Gemm.Sgemm gemm) {
        bench(gemm, g -> g.sgemm(M, N, K, 1, A, M, B, K, 0, C, M));
    }

}
