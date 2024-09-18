package org.yah.tools.gemm;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.cuda.runtime.cudaMemcpyKind;

import java.util.Objects;

public abstract class AbstractCudaGemm implements Gemm {

    protected final RuntimeAPI cuda;

    protected final StagingMemory stagingMemory = new StagingMemory();
    protected final CudaMatrix Amc;
    protected final CudaMatrix Bmc;
    protected final CudaMatrix Cmc;

    protected final Memory hostAlpha;
    protected final Memory hostBeta;

    protected AbstractCudaGemm(RuntimeAPI cuda) {
        this.cuda = Objects.requireNonNull(cuda, "cuda is null");

        Amc = new CudaMatrix();
        Bmc = new CudaMatrix();
        Cmc = new CudaMatrix();

        hostAlpha = new Memory(Double.BYTES);
        hostBeta = new Memory(Double.BYTES);
    }

    @Override
    public String toString() {
        return name();
    }

    @Override
    public void close() {
        Amc.free();
        Bmc.free();
        Cmc.free();
        stagingMemory.free();
    }

    protected final class CudaMatrix {
        private Pointer gpuPointer;
        private long currentSize;

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
            if (gpuPointer == null || currentSize < size)
                throw new IllegalStateException("invalid gpu pointer size " + (gpuPointer == null ? 0 : currentSize));
            Pointer hostPtr = stagingMemory.get(size);
            cuda.cudaMemcpy(hostPtr, gpuPointer, currentSize, cudaMemcpyKind.cudaMemcpyDeviceToHost).check();
            return hostPtr;
        }

        public Pointer allocate(long size) {
            if (size != currentSize) {
                free();
                currentSize = size;
                cuda.cudaMalloc(ptrRef, currentSize).check();
                gpuPointer = ptrRef.getValue();
            }
            return gpuPointer;
        }

        public void free() {
            if (gpuPointer != null) {
                cuda.cudaFree(gpuPointer).check();
                currentSize = 0;
                gpuPointer = null;
            }
        }

    }
}
