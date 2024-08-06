package org.yah.tools.cuda.kernel;

import org.yah.tools.cuda.runtime.cudaStream_t;
import org.yah.tools.cuda.runtime.dim3;

public class ExecutionConfig {
    private final dim3 gridDim = new dim3();
    private final dim3 blockDim = new dim3();
    private long shared;
    private cudaStream_t stream;

    public ExecutionConfig() {
    }

    public ExecutionConfig(int gridX, int gridY, int gridZ,
                           int blockX, int blockY, int blockZ,
                           long shared, cudaStream_t stream) {
        this.gridDim.set(gridX, gridY, gridZ);
        this.blockDim.set(blockX, blockY, blockZ);
        this.shared = shared;
    }

    public dim3 gridDim() {
        return gridDim;
    }

    public void gridDim(dim3 gridDim) {
        this.gridDim.set(gridDim);
    }

    public void gridDim(int x, int y, int z) {
        this.gridDim.set(x, y, z);
    }

    public dim3 blockDim() {
        return blockDim;
    }

    public void blockDim(dim3 blockDim) {
        this.blockDim.set(blockDim);
    }

    public void blockDim(int x, int y, int z) {
        this.blockDim.set(x, y, z);
    }

    public long shared() {
        return shared;
    }

    public void setShared(long shared) {
        this.shared = shared;
    }

    public cudaStream_t stream() {
        return stream;
    }

    public void setStream(cudaStream_t stream) {
        this.stream = stream;
    }

    public static ExecutionConfig singleThread() {
        return new ExecutionConfig(1, 1, 1, 1, 1, 1, 0, null);
    }
}
