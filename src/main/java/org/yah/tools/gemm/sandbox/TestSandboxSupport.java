package org.yah.tools.gemm.sandbox;

import org.yah.tools.gemm.MatrixExecutor;

import static org.yah.tools.gemm.JavaGemm.IDX2C;

public final class TestSandboxSupport {

    private final MatrixExecutor matrixExecutor;

    public TestSandboxSupport(MatrixExecutor matrixExecutor) {
        this.matrixExecutor = matrixExecutor;
    }

    @SuppressWarnings("unused")
    public void checkNaN(int M, int N, float[] matrix) {
        matrixExecutor.parallelize(M, N, (row, col) -> {
            if (Float.isNaN(matrix[IDX2C(row, col, row)]))
                throw new IllegalStateException(String.format("matrix[%d,%d] is NaN", row, col));
        });
    }

    public double maxError(int M, int N, float[] expecteds, float[] actuals) {
        double[] errors = new double[matrixExecutor.threads()];
        matrixExecutor.parallelize(M, N, threadIndex -> (row, col) -> {
            float expected = expecteds[IDX2C(row, col, M)];
            float actual = actuals[IDX2C(row, col, M)];
            float error = Math.abs(expected - actual);
            if (expected != 0) error /= expected;
            errors[threadIndex] = Math.max(error, errors[threadIndex]);
        });

        double error = 0;
        for (double e : errors) error = Math.max(e, error);
        return error;
    }

    public double maxError(int M, int N, double[] expecteds, double[] actuals) {
        double[] errors = new double[matrixExecutor.threads()];
        matrixExecutor.parallelize(M, N, threadIndex -> (row, col) -> {
            double expected = expecteds[IDX2C(row, col, M)];
            double actual = actuals[IDX2C(row, col, M)];
            double error = Math.abs(expected - actual);
            if (expected != 0) error /= expected;
            errors[threadIndex] = Math.max(error, errors[threadIndex]);
        });

        double error = 0;
        for (double e : errors) error = Math.max(e, error);
        return error;
    }

}
