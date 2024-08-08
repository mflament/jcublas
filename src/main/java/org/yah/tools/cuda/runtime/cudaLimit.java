package org.yah.tools.cuda.runtime;

import org.yah.tools.cuda.jna.NativeEnum;

/**
 * CUDA Limits
 */
@SuppressWarnings("unused")
public enum cudaLimit implements NativeEnum {

    /**
     * GPU thread stack size
     */
    cudaLimitStackSize(0),
    /**
     * GPU printf FIFO size
     */
    cudaLimitPrintfFifoSize(1),
    /**
     * GPU malloc heap size
     */
    cudaLimitMallocHeapSize(2),
    /**
     * GPU device runtime synchronize depth
     */
    cudaLimitDevRuntimeSyncDepth(3),
    /**
     * GPU device runtime pending launch count
     */
    cudaLimitDevRuntimePendingLaunchCount(4),
    /**
     * A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
     */
    cudaLimitMaxL2FetchGranularity(5),
    /**
     * A size in bytes for L2 persisting lines cache size
     */
    cudaLimitPersistingL2CacheSize(6);

    private final int value;

    cudaLimit(int value) {
        this.value = value;
    }

    @Override
    public int nativeValue() {
        return value;
    }
}
