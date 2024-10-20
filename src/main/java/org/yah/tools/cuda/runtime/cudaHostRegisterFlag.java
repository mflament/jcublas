package org.yah.tools.cuda.runtime;

public class cudaHostRegisterFlag {

    // Default host memory registration flag
    public static final int cudaHostRegisterDefault = 0x00;

    // Memory-mapped I/O space
    public static final int cudaHostRegisterIoMemory = 0x04;

    // Map registered memory into device space
    public static final int cudaHostRegisterMapped = 0x02;

    // Pinned memory accessible by all CUDA contexts
    public static final int cudaHostRegisterPortable = 0x01;

    // Memory-mapped read-only
    public static final int cudaHostRegisterReadOnly = 0x08;

}
