package org.yah.tools.cuda.runtime;

import com.sun.jna.Structure;

import java.util.List;

@SuppressWarnings("unused")
public class cudaDeviceProp extends Structure {
    /**
     * < ASCII string identifying device
     */
    public byte[] name = new byte[256];
    /**
     * < 16-byte unique identifier
     */
    public byte[] uuid = new byte[16];
    /**
     * < 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms
     */
    public byte[] luid = new byte[8];
    /**
     * < LUID device node mask. Value is undefined on TCC and non-Windows platforms
     */
    public int luidDeviceNodeMask;
    /**
     * < Global memory available on device in bytes
     */
    public long  totalGlobalMem;
    /**
     * < Shared memory available per block in bytes
     */
    public long  sharedMemPerBlock;
    /**
     * < 32-bit registers available per block
     */
    public int regsPerBlock;
    /**
     * < Warp size in threads
     */
    public int warpSize;
    /**
     * < Maximum pitch in bytes allowed by memory copies
     */
    public long  memPitch;
    /**
     * < Maximum number of threads per block
     */
    public int maxThreadsPerBlock;
    /**
     * < Maximum size of each dimension of a block
     */
    public int[] maxThreadsDim = new int[3];
    /**
     * < Maximum size of each dimension of a grid
     */
    public int[] maxGridSize = new int[3];
    /**
     * < Deprecated, Clock frequency in kilohertz
     */
    public int clockRate;
    /**
     * < Constant memory available on device in bytes
     */
    public long  totalConstMem;
    /**
     * < Major compute capability
     */
    public int major;
    /**
     * < Minor compute capability
     */
    public int minor;
    /**
     * < Alignment requirement for textures
     */
    public long  textureAlignment;
    /**
     * < Pitch alignment requirement for texture references bound to pitched memory
     */
    public long  texturePitchAlignment;
    /**
     * < Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.
     */
    public int deviceOverlap;
    /**
     * < Number of multiprocessors on device
     */
    public int multiProcessorCount;
    /**
     * < Deprecated, Specified whether there is a run time limit on kernels
     */
    public int kernelExecTimeoutEnabled;
    /**
     * < Device is integrated as opposed to discrete
     */
    public int integrated;
    /**
     * < Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
     */
    public int canMapHostMemory;
    /**
     * < Deprecated, Compute mode (See ::cudaComputeMode)
     */
    public int computeMode;
    /**
     * < Maximum 1D texture size
     */
    public int maxTexture1D;
    /**
     * < Maximum 1D mipmapped texture size
     */
    public int maxTexture1DMipmap;
    /**
     * < Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
     */
    public int maxTexture1DLinear;
    /**
     * < Maximum 2D texture dimensions
     */
    public int[] maxTexture2D = new int[2];
    /**
     * < Maximum 2D mipmapped texture dimensions
     */
    public int[] maxTexture2DMipmap = new int[2];
    /**
     * < Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
     */
    public int[] maxTexture2DLinear = new int[3];
    /**
     * < Maximum 2D texture dimensions if texture gather operations have to be performed
     */
    public int[] maxTexture2DGather = new int[2];
    /**
     * < Maximum 3D texture dimensions
     */
    public int[] maxTexture3D = new int[3];
    /**
     * < Maximum alternate 3D texture dimensions
     */
    public int[] maxTexture3DAlt = new int[3];
    /**
     * < Maximum Cubemap texture dimensions
     */
    public int maxTextureCubemap;
    /**
     * < Maximum 1D layered texture dimensions
     */
    public int[] maxTexture1DLayered = new int[2];
    /**
     * < Maximum 2D layered texture dimensions
     */
    public int[] maxTexture2DLayered = new int[3];
    /**
     * < Maximum Cubemap layered texture dimensions
     */
    public int[] maxTextureCubemapLayered = new int[2];
    /**
     * < Maximum 1D surface size
     */
    public int maxSurface1D;
    /**
     * < Maximum 2D surface dimensions
     */
    public int[] maxSurface2D = new int[2];
    /**
     * < Maximum 3D surface dimensions
     */
    public int[] maxSurface3D = new int[3];
    /**
     * < Maximum 1D layered surface dimensions
     */
    public int[] maxSurface1DLayered = new int[2];
    /**
     * < Maximum 2D layered surface dimensions
     */
    public int[] maxSurface2DLayered = new int[3];
    /**
     * < Maximum Cubemap surface dimensions
     */
    public int maxSurfaceCubemap;
    /**
     * < Maximum Cubemap layered surface dimensions
     */
    public int[] maxSurfaceCubemapLayered = new int[2];
    /**
     * < Alignment requirements for surfaces
     */
    public long  surfaceAlignment;
    /**
     * < Device can possibly execute multiple kernels concurrently
     */
    public int concurrentKernels;
    /**
     * < Device has ECC support enabled
     */
    public int ECCEnabled;
    /**
     * < PCI bus ID of the device
     */
    public int pciBusID;
    /**
     * < PCI device ID of the device
     */
    public int pciDeviceID;
    /**
     * < PCI domain ID of the device
     */
    public int pciDomainID;
    /**
     * < 1 if device is a Tesla device using TCC driver, 0 otherwise
     */
    public int tccDriver;
    /**
     * < Number of asynchronous engines
     */
    public int asyncEngineCount;
    /**
     * < Device shares a unified address space with the host
     */
    public int unifiedAddressing;
    /**
     * < Deprecated, Peak memory clock frequency in kilohertz
     */
    public int memoryClockRate;
    /**
     * < Global memory bus width in bits
     */
    public int memoryBusWidth;
    /**
     * < Size of L2 cache in bytes
     */
    public int l2CacheSize;
    /**
     * < Device's maximum l2 persisting lines capacity setting in bytes
     */
    public int persistingL2CacheMaxSize;
    /**
     * < Maximum resident threads per multiprocessor
     */
    public int maxThreadsPerMultiProcessor;
    /**
     * < Device supports stream priorities
     */
    public int streamPrioritiesSupported;
    /**
     * < Device supports caching globals in L1
     */
    public int globalL1CacheSupported;
    /**
     * < Device supports caching locals in L1
     */
    public int localL1CacheSupported;
    /**
     * < Shared memory available per multiprocessor in bytes
     */
    public long  sharedMemPerMultiprocessor;
    /**
     * < 32-bit registers available per multiprocessor
     */
    public int regsPerMultiprocessor;
    /**
     * < Device supports allocating managed memory on this system
     */
    public int managedMemory;
    /**
     * < Device is on a multi-GPU board
     */
    public int isMultiGpuBoard;
    /**
     * < Unique identifier for a group of devices on the same multi-GPU board
     */
    public int multiGpuBoardGroupID;
    /**
     * < Link between the device and the host supports native atomic operations
     */
    public int hostNativeAtomicSupported;
    /**
     * < Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance
     */
    public int singleToDoublePrecisionPerfRatio;
    /**
     * < Device supports coherently accessing pageable memory without calling cudaHostRegister on it
     */
    public int pageableMemoryAccess;
    /**
     * < Device can coherently access managed memory concurrently with the CPU
     */
    public int concurrentManagedAccess;
    /**
     * < Device supports Compute Preemption
     */
    public int computePreemptionSupported;
    /**
     * < Device can access host registered memory at the same virtual address as the CPU
     */
    public int canUseHostPointerForRegisteredMem;
    /**
     * < Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
     */
    public int cooperativeLaunch;
    /**
     * < Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.
     */
    public int cooperativeMultiDeviceLaunch;
    /**
     * < Per device maximum shared memory per block usable by special opt in
     */
    public long  sharedMemPerBlockOptin;
    /**
     * < Device accesses pageable memory via the host's page tables
     */
    public int pageableMemoryAccessUsesHostPageTables;
    /**
     * < Host can directly access managed memory on the device without migration.
     */
    public int directManagedMemAccessFromHost;
    /**
     * < Maximum number of resident blocks per multiprocessor
     */
    public int maxBlocksPerMultiProcessor;
    /**
     * < The maximum value of ::cudaAccessPolicyWindow::num_bytes.
     */
    public int accessPolicyMaxWindowSize;
    /**
     * < Shared memory reserved by CUDA driver per block in bytes
     */
    public long  reservedSharedMemPerBlock;
    /**
     * < Device supports host memory registration via ::cudaHostRegister.
     */
    public int hostRegisterSupported;
    /**
     * < 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise
     */
    public int sparseCudaArraySupported;
    /**
     * < Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU
     */
    public int hostRegisterReadOnlySupported;
    /**
     * < External timeline semaphore interop is supported on the device
     */
    public int timelineSemaphoreInteropSupported;
    /**
     * < 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise
     */
    public int memoryPoolsSupported;
    /**
     * < 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
     */
    public int gpuDirectRDMASupported;
    /**
     * < Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum
     */
    public int gpuDirectRDMAFlushWritesOptions;
    /**
     * < See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values
     */
    public int gpuDirectRDMAWritesOrdering;
    /**
     * < Bitmask of handle types supported with mempool-based IPC
     */
    public int memoryPoolSupportedHandleTypes;
    /**
     * < 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
     */
    public int deferredMappingCudaArraySupported;
    /**
     * < Device supports IPC Events.
     */
    public int ipcEventSupported;
    /**
     * < Indicates device supports cluster launch
     */
    public int clusterLaunch;
    /**
     * < Indicates device supports unified pointers
     */
    public int unifiedFunctionPointers;
    public int[] reserved2 = new int[2];
    /**
     * < Reserved for future use
     */
    public int[] reserved1 = new int[1];
    /**
     * < Reserved for future use
     */
    public int[] reserved = new int[60];

    public static class ByReference extends cudaDeviceProp implements Structure.ByReference {
    }

    @Override
    protected List<String> getFieldOrder() {
        return FIELDS;
    }

    static final List<String> FIELDS = List.of(
            "name",
            "uuid",
            "luid",
            "luidDeviceNodeMask",
            "totalGlobalMem",
            "sharedMemPerBlock",
            "regsPerBlock",
            "warpSize",
            "memPitch",
            "maxThreadsPerBlock",
            "maxThreadsDim",
            "maxGridSize",
            "clockRate",
            "totalConstMem",
            "major",
            "minor",
            "textureAlignment",
            "texturePitchAlignment",
            "deviceOverlap",
            "multiProcessorCount",
            "kernelExecTimeoutEnabled",
            "integrated",
            "canMapHostMemory",
            "computeMode",
            "maxTexture1D",
            "maxTexture1DMipmap",
            "maxTexture1DLinear",
            "maxTexture2D",
            "maxTexture2DMipmap",
            "maxTexture2DLinear",
            "maxTexture2DGather",
            "maxTexture3D",
            "maxTexture3DAlt",
            "maxTextureCubemap",
            "maxTexture1DLayered",
            "maxTexture2DLayered",
            "maxTextureCubemapLayered",
            "maxSurface1D",
            "maxSurface2D",
            "maxSurface3D",
            "maxSurface1DLayered",
            "maxSurface2DLayered",
            "maxSurfaceCubemap",
            "maxSurfaceCubemapLayered",
            "surfaceAlignment",
            "concurrentKernels",
            "ECCEnabled",
            "pciBusID",
            "pciDeviceID",
            "pciDomainID",
            "tccDriver",
            "asyncEngineCount",
            "unifiedAddressing",
            "memoryClockRate",
            "memoryBusWidth",
            "l2CacheSize",
            "persistingL2CacheMaxSize",
            "maxThreadsPerMultiProcessor",
            "streamPrioritiesSupported",
            "globalL1CacheSupported",
            "localL1CacheSupported",
            "sharedMemPerMultiprocessor",
            "regsPerMultiprocessor",
            "managedMemory",
            "isMultiGpuBoard",
            "multiGpuBoardGroupID",
            "hostNativeAtomicSupported",
            "singleToDoublePrecisionPerfRatio",
            "pageableMemoryAccess",
            "concurrentManagedAccess",
            "computePreemptionSupported",
            "canUseHostPointerForRegisteredMem",
            "cooperativeLaunch",
            "cooperativeMultiDeviceLaunch",
            "sharedMemPerBlockOptin",
            "pageableMemoryAccessUsesHostPageTables",
            "directManagedMemAccessFromHost",
            "maxBlocksPerMultiProcessor",
            "accessPolicyMaxWindowSize",
            "reservedSharedMemPerBlock",
            "hostRegisterSupported",
            "sparseCudaArraySupported",
            "hostRegisterReadOnlySupported",
            "timelineSemaphoreInteropSupported",
            "memoryPoolsSupported",
            "gpuDirectRDMASupported",
            "gpuDirectRDMAFlushWritesOptions",
            "gpuDirectRDMAWritesOrdering",
            "memoryPoolSupportedHandleTypes",
            "deferredMappingCudaArraySupported",
            "ipcEventSupported",
            "clusterLaunch",
            "unifiedFunctionPointers",
            "reserved2",
            "reserved1",
            "reserved"
    );
}
