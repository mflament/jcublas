package org.yah.tools.gemm;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.cublas.*;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.gemm.Times.Operation;

import javax.annotation.Nullable;
import java.util.Objects;

public abstract class CublasGemm extends AbstractCudaGemm {

    private static final Logger LOGGER = LoggerFactory.getLogger(CublasGemm.class);

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

    protected final CublasAPI cublas;
    protected final cublasHandle_t handle;

    protected GemmConfig config;
    protected final Times times;

    protected CublasGemm(RuntimeAPI cuda, CublasAPI cublas) {
        super(cuda);
        times = new Times(this.name());
        this.cublas = Objects.requireNonNull(cublas, "cublas is null");
        this.handle = createCublasHandle(cublas);
        cublas.cublasSetPointerMode(handle, cublasPointerMode_t.CUBLAS_POINTER_MODE_HOST).check();
    }

    @Override
    public String name() {
        return "cublas";
    }

    @Override
    public Times times() {
        return times;
    }

    @Override
    public GemmId id() {
        return GemmId.CUBLAS;
    }

    private static cublasHandle_t createCublasHandle(CublasAPI cublas) {
        cublasHandle_t.ByReference handleRef = new cublasHandle_t.ByReference();
        cublas.cublasCreate(handleRef).check();
        cublasHandle_t handle = handleRef.getValue();

        IntByReference versionRef = new IntByReference();
        cublas.cublasGetVersion(handle, versionRef).check();
        int version = versionRef.getValue();
        int major = version / 10000;
        int minor = (version - major * 10000) / 100;
        int subminor = version - major * 10000 - minor * 100;
        LOGGER.debug("cublas version {}.{}.{}", major, minor, subminor);
        return handle;
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
        super.close();
        cublas.cublasDestroy(handle);
    }

    public static final class CublasSgemm extends CublasGemm implements Sgemm {

        public CublasSgemm(RuntimeAPI cuda, CublasAPI cublas) {
            super(cuda, cublas);
        }

        @Override
        public GemmId id() {
            return GemmId.CUBLAS;
        }

        @Override
        public void sgemm(int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc) {
            long start = System.nanoTime();
            Pointer gpuA = Amc.write(M, K, A);
            Pointer gpuB = Bmc.write(K, N, B);
            Pointer gpuC = beta == 0 ? Cmc.allocate(M * N * (long) Float.BYTES) : Cmc.write(M, N, C);
            hostAlpha.setFloat(0, alpha);
            hostBeta.setFloat(0, beta);
            times.addNanos(Operation.WRITE, System.nanoTime() - start);

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
            times.addNanos(Operation.GEMM, System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos(Operation.READ, System.nanoTime() - start);
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
            times.addNanos(Operation.WRITE, System.nanoTime() - start);

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
            times.addNanos(Operation.GEMM, System.nanoTime() - start);

            start = System.nanoTime();
            Cmc.read(M, N, C);
            times.addNanos(Operation.READ, System.nanoTime() - start);
        }
    }

}
