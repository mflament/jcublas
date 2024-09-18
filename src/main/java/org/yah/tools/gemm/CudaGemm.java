package org.yah.tools.gemm;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.yah.tools.cuda.driver.CUresult;
import org.yah.tools.cuda.kernel.ExecutionConfig;
import org.yah.tools.cuda.kernel.KernelSupport;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.cuda.runtime.dim3;
import org.yah.tools.gemm.Times.Operation;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.yah.tools.gemm.MatrixExecutor.ceil_div;

public abstract class CudaGemm extends AbstractCudaGemm {

    public static final dim3 CUDA_BLOCK_SIZE = new dim3(8, 128, 1);

    public interface GemmKernels {
        CUresult sgemm(ExecutionConfig executionConfig,
                       int M, int N, int K,
                       float alpha, Pointer A, int lda,
                       Pointer B, int ldb,
                       float beta,
                       Pointer C, int ldc);

        CUresult dgemm(ExecutionConfig executionConfig,
                       int M, int N, int K,
                       double alpha, Pointer A, int lda,
                       Pointer B, int ldb,
                       double beta,
                       Pointer C, int ldc);

    }

    private static final int TILE_SIZE = 32;

    private final GemmId id;
    private final String name;

    private final KernelSupport kernelSupport;
    protected final Pointer cuContext;
    protected final Pointer cuModule;
    protected final Times times;
    protected GemmKernels gemmKernels;

    protected CudaGemm(RuntimeAPI cuda, int deviceOrdinal, String sourcePath, Map<String, String> defines,
                       boolean tiled, boolean transposed) {
        super(cuda);
        this.id = getId(tiled, transposed);
        this.name = getName(tiled, transposed);
        this.times = new Times(name);
        kernelSupport = new KernelSupport(deviceOrdinal);
        Pointer program = kernelSupport.compile(loadSource(sourcePath), defines);
        Memory ptx = kernelSupport.getPTX(program);
//        System.out.println(ptx.getString(0, "US-ASCII"));
        kernelSupport.destroyProgram(program);

        cuContext = kernelSupport.getPrimaryContext();
        cuModule = kernelSupport.loadModule(cuContext, ptx);
        ptx.close();

        gemmKernels = kernelSupport.createProxy(cuModule, GemmKernels.class);
    }

    @Override
    public GemmId id() {
        return id;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Times times() {
        return times;
    }

    protected final GemmId getId(boolean tiled, boolean transposed) {
        if (tiled && transposed)
            return GemmId.CUDA_TILED_TRANSPOSED;
        if (tiled)
            return GemmId.CUDA_TILED;
        return GemmId.CUDA;
    }

    protected final String getName(boolean tiled, boolean transposed) {
        if (transposed)
            return "cuda+tiled+TR";
        if (tiled)
            return "cuda+tiled";
        return "cuda";
    }

    @Override
    public void close() {
        super.close();
        kernelSupport.unloadModule(cuModule);
        kernelSupport.releasePrimaryContext();
    }

    public static class CudaSgemm extends CudaGemm implements Sgemm {
        private final MatrixExecutor matrixExecutor;
        private final ExecutionConfig executionConfig = new ExecutionConfig();
        private final dim3 blockDim;
        private final boolean transpose;
        private float[] transposedA;

        public CudaSgemm(RuntimeAPI cuda, MatrixExecutor matrixExecutor, int deviceOrdinal, boolean tiled, boolean transposed) {
            super(cuda, deviceOrdinal, transposed ? "cuda/transposed_gemm.cu" : tiled ? "cuda/tiled_gemm.cu" : "cuda/gemm.cu",
                    Map.of("BLOCK_SIZE", Integer.toString(TILE_SIZE)), tiled, transposed);
            this.matrixExecutor = matrixExecutor;
            transpose = transposed;
            if (transpose) this.blockDim = new dim3(TILE_SIZE, TILE_SIZE, 1);
            else if (tiled) this.blockDim = new dim3(TILE_SIZE, TILE_SIZE, 1);
            else this.blockDim = CUDA_BLOCK_SIZE;
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            int tlda = lda;
            float[] tA = A;
            if (transpose) {
                times.measure(Operation.TRANSPOSE, () -> {
                    if (transposedA == null || transposedA.length < A.length)
                        transposedA = new float[A.length];
                    JavaGemm.transpose(matrixExecutor, M, K, transposedA, A, lda);
                });
                tlda = K;
                tA = transposedA;
            }

            long start = System.nanoTime();
            Pointer gpuA = Amc.write(M, K, tA);
            Pointer gpuB = Bmc.write(K, N, B);
            Pointer gpuC = beta == 0 ? Cmc.allocate(M * N * (long) Float.BYTES) : Cmc.write(M, N, C);
            hostAlpha.setFloat(0, alpha);
            hostBeta.setFloat(0, beta);
            times.addNanos(Operation.WRITE, System.nanoTime() - start);

            executionConfig.blockDim(blockDim);
            executionConfig.gridDim(ceil_div(N, blockDim.x), ceil_div(M, blockDim.y), 1);
//            System.out.printf("gridDim=%s blockDim=%s%n", executionConfig.gridDim(), blockDim);
            start = System.nanoTime();
            gemmKernels.sgemm(executionConfig, M, N, K, alpha, gpuA, tlda, gpuB, ldb, beta, gpuC, ldc).check();
            cuda.cudaDeviceSynchronize().check();
            times.addNanos(Operation.GEMM, System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos(Operation.READ, System.nanoTime() - start);
        }
    }

    public static class CudaDgemm extends CudaGemm implements Dgemm {
        private final MatrixExecutor matrixExecutor;
        private final ExecutionConfig executionConfig = new ExecutionConfig();
        private final dim3 blockDim;
        private final boolean transpose;
        private double[] transposedA;

        public CudaDgemm(RuntimeAPI cuda, MatrixExecutor matrixExecutor, int deviceOrdinal, boolean tiled, boolean transposed) {
            super(cuda, deviceOrdinal, transposed ? "cuda/transposed_gemm.cu" : tiled ? "cuda/tiled_gemm.cu" : "cuda/gemm.cu",
                    Map.of("BLOCK_SIZE", Integer.toString(TILE_SIZE)), tiled, transposed);
            this.matrixExecutor = matrixExecutor;
            transpose = transposed;
            if (transpose) this.blockDim = new dim3(TILE_SIZE, TILE_SIZE, 1);
            else if (tiled) this.blockDim = new dim3(TILE_SIZE, TILE_SIZE, 1);
            else this.blockDim = CUDA_BLOCK_SIZE;
        }

        @Override
        public void dgemm(int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc) {
            int tlda = lda;
            double[] tA = A;
            if (transpose) {
                times.measure(Operation.TRANSPOSE, () -> {
                    if (transposedA == null || transposedA.length < A.length)
                        transposedA = new double[A.length];
                    JavaGemm.transpose(matrixExecutor, M, K, transposedA, A, lda);
                });
                tlda = K;
                tA = transposedA;
            }

            long start = System.nanoTime();
            Pointer gpuA = Amc.write(M, K, tA);
            Pointer gpuB = Bmc.write(K, N, B);
            Pointer gpuC = beta == 0 ? Cmc.allocate(M * N * (long) Double.BYTES) : Cmc.write(M, N, C);
            hostAlpha.setDouble(0, alpha);
            hostBeta.setDouble(0, beta);
            times.addNanos(Operation.WRITE, System.nanoTime() - start);

            executionConfig.blockDim(blockDim);
            executionConfig.gridDim(ceil_div(N, blockDim.x), ceil_div(M, blockDim.y), 1);
//            System.out.printf("gridDim=%s blockDim=%s%n", executionConfig.gridDim(), blockDim);
            start = System.nanoTime();
            gemmKernels.dgemm(executionConfig, M, N, K, alpha, gpuA, tlda, gpuB, ldb, beta, gpuC, ldc).check();
            cuda.cudaDeviceSynchronize().check();
            times.addNanos(Operation.GEMM, System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos(Operation.READ, System.nanoTime() - start);
        }
    }

    public static String loadSource(String resource) {
        try (InputStream is = ClassLoader.getSystemClassLoader().getResourceAsStream(resource)) {
            if (is == null)
                throw new FileNotFoundException("classpath resource '" + resource + "'");
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}
