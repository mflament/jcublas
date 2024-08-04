package og.yah.tools.gemm;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import og.yah.tools.cuda.cublas.*;
import og.yah.tools.cuda.jna.size_t;
import og.yah.tools.cuda.runtime.RuntimeAPI;
import og.yah.tools.cuda.runtime.cudaMemcpyKind;

import javax.annotation.Nullable;
import java.util.Objects;

public abstract class CublasGemm implements Gemm {

    public static class GemmConfig {
        private final cublasAtomicsMode_t atomicMode;
        private final int mathMode;
        private final cublasComputeType_t computeType;
        private final cublasGemmAlgo_t algo;

        public GemmConfig(cublasAtomicsMode_t atomicMode, int mathMode, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
            this.atomicMode = atomicMode;
            this.mathMode = mathMode;
            this.computeType = computeType;
            this.algo = algo;
        }
    }

    protected final RuntimeAPI cuda;
    protected final CublasAPI cublas;

    protected final cublasHandle_t handle;

    protected final StagingMemory stagingMemory = new StagingMemory();
    protected final MatrixCache Amc = new MatrixCache();
    protected final MatrixCache Bmc = new MatrixCache();
    protected final MatrixCache Cmc = new MatrixCache();

    protected final Memory hostAlpha;
    protected final Memory hostBeta;

    protected GemmConfig config;

    protected final Times times = new Times();

    protected CublasGemm(RuntimeAPI cuda, CublasAPI cublas) {
        this.cuda = Objects.requireNonNull(cuda, "cuda is null");
        this.cublas = Objects.requireNonNull(cublas, "cublas is null");

        cublasHandle_t.ByReference handleRef = new cublasHandle_t.ByReference();
        cublas.cublasCreate(handleRef).check();
        handle = handleRef.getValue();

        IntByReference versionRef = new IntByReference();
        cublas.cublasGetVersion(handle, versionRef).check();
        int version = versionRef.getValue();
        int major = version / 10000;
        int minor = (version - major * 10000) / 100;
        int subminor = version - major * 10000 - minor * 100;
        System.out.printf("cublas version %d.%d.%d%n", major, minor, subminor);

        cublas.cublasSetPointerMode(handle, cublasPointerMode_t.CUBLAS_POINTER_MODE_HOST).check();

        hostAlpha = new Memory(Double.BYTES);
        hostBeta = new Memory(Double.BYTES);
    }

    @Override
    public final Times times() {
        return times;
    }

    @Override
    public String toString() {
        return name();
    }

    public final void setConfig(@Nullable GemmConfig config) {
        this.config = config;
        if (config != null) {
            cublas.cublasSetAtomicsMode(handle, config.atomicMode).check();
            cublas.cublasSetMathMode(handle, config.mathMode).check();
        } else {
            cublas.cublasSetAtomicsMode(handle, cublasAtomicsMode_t.CUBLAS_ATOMICS_ALLOWED).check();
            cublas.cublasSetMathMode(handle, cublasMath_t.CUBLAS_DEFAULT_MATH).check();
        }
    }

    @Override
    public final void close() {
        Amc.free();
        Bmc.free();
        Cmc.free();
    }

    public static final class CublasSgemm extends CublasGemm implements Sgemm {

        public CublasSgemm(RuntimeAPI cuda, CublasAPI cublas) {
            super(cuda, cublas);
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            long start = System.nanoTime();
            Pointer gpuA = Amc.write(M, K, A);
            Pointer gpuB = Bmc.write(K, N, B);
            Pointer gpuC = beta == 0 ? Cmc.allocate(M * N * (long) Float.BYTES) : Cmc.write(M, N, C);
            hostAlpha.setFloat(0, alpha);
            hostBeta.setFloat(0, beta);
            times.addNanos("write", System.nanoTime() - start);

            start = System.nanoTime();
            if (config != null) {
                cublas.cublasGemmEx(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, M, N, K,
                        hostAlpha, gpuA, cudaDataType_t.CUDA_R_32F, lda,
                        gpuB, cudaDataType_t.CUDA_R_32F, ldb,
                        hostBeta, gpuC, cudaDataType_t.CUDA_R_32F, ldc,
                        config.computeType, config.algo).check();
            } else {
                cublas.cublasSgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, M, N, K,
                        hostAlpha, gpuA, lda,
                        gpuB, ldb,
                        hostBeta, gpuC, ldc).check();
            }
            cuda.cudaDeviceSynchronize().check();
            times.addNanos("sgemm", System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos("read", System.nanoTime() - start);
        }
    }


    public static final class CublasDgemm extends CublasGemm implements Dgemm {

        public CublasDgemm(RuntimeAPI cuda, CublasAPI cublas) {
            super(cuda, cublas);
        }

        @Override
        public void dgemm(int M, int N, int K, double alpha, double[] A, int lda, double[] B, int ldb, double beta, double[] C, int ldc) {
            long start = System.nanoTime();
            Pointer gpuA = Amc.write(M, K, A);
            Pointer gpuB = Bmc.write(K, N, B);
            Pointer gpuC = beta == 0 ? Cmc.allocate(M * N * (long) Double.BYTES) : Cmc.write(M, N, C);
            hostAlpha.setDouble(0, alpha);
            hostBeta.setDouble(0, beta);
            times.addNanos("write", System.nanoTime() - start);

            start = System.nanoTime();
            if (config != null) {
                cublas.cublasGemmEx(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, M, N, K,
                        hostAlpha, gpuA, cudaDataType_t.CUDA_R_64F, lda,
                        gpuB, cudaDataType_t.CUDA_R_64F, ldb,
                        hostBeta, gpuC, cudaDataType_t.CUDA_R_64F, ldc,
                        config.computeType, config.algo).check();
            } else {
                cublas.cublasDgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, M, N, K,
                        hostAlpha, gpuA, lda,
                        gpuB, ldb,
                        hostBeta, gpuC, ldc).check();
            }
            cuda.cudaDeviceSynchronize().check();
            times.addNanos("dgemm", System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos("read", System.nanoTime() - start);
        }
    }

    protected final class MatrixCache {
        private Pointer gpuPointer;
        private final size_t currentSize = new size_t();

        private final PointerByReference ptrRef = new PointerByReference();

        public Pointer write(int row, int col, float[] data) {
            long size = row * col * (long) Float.BYTES;
            Pointer hostPtr = stagingMemory.get(size);
            hostPtr.write(0, data, 0, data.length);
            return write(hostPtr, size);
        }

        public void read(int row, int col, float[] data) {
            long size = row * col * (long) Float.BYTES;
            read(size).read(0, data, 0, data.length);
        }

        public Pointer write(int row, int col, double[] data) {
            long size = row * col * (long) Double.BYTES;
            Pointer hostPtr = stagingMemory.get(size);
            hostPtr.write(0, data, 0, data.length);
            return write(hostPtr, size);
        }

        public void read(int row, int col, double[] data) {
            long size = row * col * (long) Double.BYTES;
            read(size).read(0, data, 0, data.length);
        }

        private Pointer write(Pointer hostPtr, long size) {
            Pointer gpuPtr = allocate(size);
            cuda.cudaMemcpy(gpuPtr, hostPtr, currentSize, cudaMemcpyKind.cudaMemcpyHostToDevice).check();
            return gpuPtr;
        }

        private Pointer read(long size) {
            if (gpuPointer == null || currentSize.getValue() < size)
                throw new IllegalStateException("invalid gpu pointer size " + (gpuPointer == null ? 0 : currentSize.toString()));
            Pointer hostPtr = stagingMemory.get(size);
            cuda.cudaMemcpy(hostPtr, gpuPointer, currentSize, cudaMemcpyKind.cudaMemcpyDeviceToHost).check();
            return hostPtr;
        }

        public Pointer allocate(long size) {
            if (size != currentSize.getValue()) {
                free();
                currentSize.setValue(size);
                cuda.cudaMalloc(ptrRef, currentSize).check();
                gpuPointer = ptrRef.getValue();
            }
            return gpuPointer;
        }

        public void free() {
            if (gpuPointer != null) {
                cuda.cudaFree(gpuPointer).check();
                currentSize.setValue(0);
                gpuPointer = null;
            }
            stagingMemory.free();
        }

    }

}
