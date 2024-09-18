package org.yah.tools.gemm.sandbox;

import com.sun.jna.Native;
import org.yah.tools.cuda.cublas.CublasAPI;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.cuda.runtime.cudaDeviceProp;
import org.yah.tools.gemm.CublasGemm.CublasSgemm;
import org.yah.tools.gemm.Gemm;
import org.yah.tools.gemm.GemmId;
import org.yah.tools.gemm.JavaGemm;
import org.yah.tools.gemm.MatrixExecutor;
import org.yah.tools.gemm.Times;
import org.yah.tools.gemm.Times.Operation;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static org.yah.tools.gemm.CublasGemm.CublasDgemm;
import static org.yah.tools.gemm.CudaGemm.CudaDgemm;
import static org.yah.tools.gemm.CudaGemm.CudaSgemm;

public abstract class AbstractGemmSandbox {

    protected static final int warmup = 1;
    protected static final int runs = 3;
    protected static final Set<GemmId> benchedGemms = EnumSet.of(
            GemmId.ST,
            GemmId.MT,
            GemmId.MT_TRANSPOSED,
            GemmId.CUBLAS,
            GemmId.CUDA,
            GemmId.CUDA_TILED
            //GemmId.CUDA_TILED_TRANSPOSED
    );

    public static final int MAX_ARRAY_LENGTH = Integer.MAX_VALUE - 2;
    protected static RuntimeAPI cuda;
    protected static CublasAPI cublas;
    protected static MatrixExecutor matrixExecutor;

    private static final double MB = 1024 * 1024;

    protected int cudaDevice = 0;
    protected int M = 2000;
    protected int N = 1500;
    protected int K = 3000;

    private final List<Times> benchTimes = new ArrayList<>();

    protected AbstractGemmSandbox() {
    }

    protected final void execute(String[] args) {
        parseCommandLine(args);
        setup();
        try {
            run();
        } finally {
            tearDown();
        }
    }

    protected void parseCommandLine(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-M" -> M = Integer.parseInt(args[++i]);
                case "-N" -> N = Integer.parseInt(args[++i]);
                case "-K" -> K = Integer.parseInt(args[++i]);
                default -> {
                    System.err.println("Invalid parameter " + args[i]);
                    System.exit(1);
                }
            }
        }
    }

    protected void setup() {
        validateDims();
        System.out.printf("type=%s A=%s B=%s C=%s%n", elementType(), matrixInfo(M, K), matrixInfo(K, N), matrixInfo(M, N));
        cuda = RuntimeAPI.load();
        cublas = CublasAPI.load();
        matrixExecutor = new MatrixExecutor();
        cuda.cudaSetDevice(cudaDevice);

        cudaDeviceProp.ByReference props = new cudaDeviceProp.ByReference();
        cuda.cudaGetDeviceProperties(props, cudaDevice).check();
        System.out.printf("Cuda device %s (SM=%d, threads/SM=%d, threads=%s, singleToDoublePrecisionPerfRatio=%d)%n", Native.toString(props.name),
                props.multiProcessorCount, props.maxThreadsPerMultiProcessor,
                props.multiProcessorCount * props.maxThreadsPerMultiProcessor,
                props.singleToDoublePrecisionPerfRatio);
    }

    protected abstract String elementType();

    private String matrixInfo(int rows, int cols) {
        return String.format("%dx%d (%.2fMB)", rows, cols, rows * cols * elementSize() / MB);
    }

    private void validateDims() {
        validateDim("A", M * K);
        validateDim("B", K * N);
        validateDim("C", M * N);
    }

    private void validateDim(String matrix, int dim) {
        if (dim > MAX_ARRAY_LENGTH) // max array dim
            throw new IllegalStateException(matrix + " array length " + dim + " overflow max length " + MAX_ARRAY_LENGTH);
    }

    protected void tearDown() {
        if (matrixExecutor != null) matrixExecutor.close();
    }

    protected abstract void run();

    protected static MatrixExecutor.Handler randomizer(long seed, float[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[JavaGemm.IDX2C(row, col, ld)] = random.nextFloat();
    }

    protected static MatrixExecutor.Handler randomizer(long seed, double[] matrix, int ld) {
        Random random = new Random(seed);
        return (row, col) -> matrix[JavaGemm.IDX2C(row, col, ld)] = random.nextDouble();
    }

    protected abstract long elementSize();

    protected final <G extends Gemm> void bench(G gemm, Consumer<G> caller) {
        if (!benchedGemms.contains(gemm.id()))
            return;

        System.out.printf("%-15s: ", gemm.name());

        for (int i = 0; i < warmup; i++) caller.accept(gemm);
        gemm.times().reset();

        for (int i = 0; i < runs; i++) caller.accept(gemm);
        benchTimes.add(new Times(gemm.times()));
        System.out.println("done");
    }

    protected final String formatTimes() {
        StringBuilder sb = new StringBuilder();
        Operation[] operations = Operation.values();
        int columns = 1 + operations.length + 1;
        int nameSize = 15, colSize = 10;
        int width = (nameSize + 1) + operations.length * (colSize + 1) + colSize + 1 + columns + 1;
        String separator = "-".repeat(width) + "\n";
        sb.append(separator);
        append(sb, "| %-" + nameSize + "s|", "name");
        for (Operation operation : operations)
            append(sb, " %-" + colSize + "s|", operation);
        append(sb, " %-" + colSize + "s|", "total");
        sb.append("\n");

        sb.append(separator);
        for (Times benchTime : benchTimes) {
            append(sb, "| %-" + nameSize + "s|", benchTime.name());
            for (Operation operation : operations)
                append(sb, " %s|", formatTime(benchTime.get(operation), colSize));
            append(sb, " %s|", formatTime(benchTime.total(), colSize));
            sb.append("\n");
        }
        sb.append(separator);
        return sb.toString();
    }

    protected final void exportCSV(String csvFile) {
        String separator = ",";
        StringBuilder sb = new StringBuilder();
        Operation[] operations = Operation.values();
        sb.append("name").append(separator);
        sb.append(Arrays.stream(operations).map(Operation::toString).collect(Collectors.joining(separator))).append(separator);
        sb.append("total").append("\n");

        for (Times benchTime : benchTimes) {
            sb.append(benchTime.name()).append(separator);
            for (Operation operation : operations) {
                sb.append(String.format(Locale.US,"%.1f", benchTime.get(operation))).append(separator);
            }
            sb.append(String.format(Locale.US,"%.1f", benchTime.total())).append(separator).append("\n");
        }

        try {
            Files.writeString(Paths.get(csvFile), sb.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    protected static String formatTime(double time, int width) {
        return String.format(Locale.US,"%-" + width + ".1f", time);
    }

    private static void append(StringBuilder sb, String format, Object... args) {
        sb.append(String.format(format, args));
    }

    public static abstract class AbstractSgemmSandbox extends AbstractGemmSandbox {
        protected static Gemm.Sgemm singleThreadGemm;
        protected static Gemm.Sgemm parallelizedGemm;
        protected static Gemm.Sgemm parallelizedTransposedSgemm;
        protected static CublasSgemm cublasGemm;
        protected static CudaSgemm cudaGemm;
        protected static CudaSgemm cudaTiledGemm;
        protected static CudaSgemm cudaTransposedGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new JavaGemm.SingleThreadSgemm();
            parallelizedGemm = new JavaGemm.ParallelizedSgemm(matrixExecutor);
            parallelizedTransposedSgemm = new JavaGemm.ParallelizedTransposedSgemm(matrixExecutor);
            cublasGemm = new CublasSgemm(cuda, cublas);
            cudaGemm = new CudaSgemm(cuda, matrixExecutor, cudaDevice, false, false);
            cudaTiledGemm = new CudaSgemm(cuda, matrixExecutor, cudaDevice, true, false);
            cudaTransposedGemm = new CudaSgemm(cuda, matrixExecutor, cudaDevice, false, true);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
            if (cudaGemm != null) cudaGemm.close();
            if (cudaTiledGemm != null) cudaTiledGemm.close();
            if (cudaTransposedGemm != null) cudaTransposedGemm.close();
        }

        @Override
        protected String elementType() {
            return "float";
        }

        @Override
        protected final long elementSize() {
            return Float.BYTES;
        }
    }

    public static abstract class AbstractDgemmSandbox extends AbstractGemmSandbox {
        protected static Gemm.Dgemm singleThreadGemm;
        protected static Gemm.Dgemm parallelizedGemm;
        protected static Gemm.Dgemm parallelizedTransposedSgemm;
        protected static CublasDgemm cublasGemm;
        protected static CudaDgemm cudaGemm;
        protected static CudaDgemm cudaTiledGemm;
        protected static CudaDgemm cudaTransposedGemm;

        @Override
        protected void setup() {
            super.setup();
            singleThreadGemm = new JavaGemm.SingleThreadDgemm();
            parallelizedGemm = new JavaGemm.ParallelizedDgemm(matrixExecutor);
            parallelizedTransposedSgemm = new JavaGemm.ParallelizedTransposedDgemm(matrixExecutor);
            cublasGemm = new CublasDgemm(cuda, cublas);
            cudaGemm = new CudaDgemm(cuda, matrixExecutor, cudaDevice, false, false);
            cudaTiledGemm = new CudaDgemm(cuda, matrixExecutor, cudaDevice, true, false);
            cudaTransposedGemm = new CudaDgemm(cuda, matrixExecutor, cudaDevice, false, true);
        }

        @Override
        protected void tearDown() {
            super.tearDown();
            if (singleThreadGemm != null) singleThreadGemm.close();
            if (parallelizedGemm != null) parallelizedGemm.close();
            if (parallelizedTransposedSgemm != null) parallelizedTransposedSgemm.close();
            if (cublasGemm != null) cublasGemm.close();
            if (cudaGemm != null) cudaGemm.close();
            if (cudaTiledGemm != null) cudaTiledGemm.close();
            if (cudaTransposedGemm != null) cudaTransposedGemm.close();
        }

        @Override
        protected String elementType() {
            return "double";
        }

        @Override
        protected final long elementSize() {
            return Double.BYTES;
        }
    }
}
