package org.yah.tools.memcpy;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.driver.CUdevice_attribute;
import org.yah.tools.cuda.driver.CUresult;
import org.yah.tools.cuda.driver.DriverAPI;
import org.yah.tools.cuda.kernel.ExecutionConfig;
import org.yah.tools.cuda.kernel.KernelSupport;
import org.yah.tools.cuda.kernel.OccupancyHelper;
import org.yah.tools.cuda.nvrtc.NVRTC;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.cuda.runtime.cudaHostRegisterFlag;

import java.util.Map;

public class MemcpyKernelsSupport implements AutoCloseable {

    // @formatter:off
    public static final String KERNELS = """
typedef unsigned int uint32;
typedef long long int64;
extern "C" __global__ void scale(int64* data, uint32 length, int factor)
{
    for (uint32 x = blockDim.x * blockIdx.x + threadIdx.x; x < length; x += gridDim.x * blockDim.x)
    {
        data[x] *= factor;
    }
}

extern "C" __global__ void memcpy(int64* dst, int64* src, uint32 length)
{
    for (uint32 x = blockDim.x * blockIdx.x + threadIdx.x; x < length; x += gridDim.x * blockDim.x)
    {
        dst[x] = src[x];
    }
}
            """; // @formatter:on

    public interface MemcpyKernels {
        /**
         * data = data * scale
         */
        CUresult scale(ExecutionConfig executionConfig, Pointer data, int length, int factor);

        CUresult memcpy(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);
    }

    private final RuntimeAPI cuda;
    private final DriverAPI cu;

    private final KernelSupport kernelSupport;
    private final Pointer cuModule;
    private final MemcpyKernels kernels;
    private final ExecutionConfig scaleExecutionConfig;
    private final ExecutionConfig memcpyExecutionConfig;

    public MemcpyKernelsSupport(RuntimeAPI cuda, int deviceOrdinal, int blockSize) {
        this.cuda = cuda;
        this.cu = DriverAPI.load();

        kernelSupport = new KernelSupport(deviceOrdinal, cu, NVRTC.load());
        Pointer program = kernelSupport.compile(KERNELS, Map.of());
        try (Memory ptx = kernelSupport.getPTX(program)) {
//        System.out.println(ptx.getString(0, "US-ASCII"));
            kernelSupport.destroyProgram(program);
            Pointer cuContext = kernelSupport.getPrimaryContext();
            cuModule = kernelSupport.loadModule(cuContext, ptx);
        }
        kernels = kernelSupport.createProxy(cuModule, MemcpyKernels.class);

        PointerByReference deviceRef = new PointerByReference();
        cu.cuDeviceGet(deviceRef, deviceOrdinal).check();
        IntByReference smCountRef = new IntByReference();
        cu.cuDeviceGetAttribute(smCountRef, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceRef.getValue()).check();

        int smCount = smCountRef.getValue();
        int maxBlockParSM = OccupancyHelper.getMaxActiveBlocksPerMultiprocessor(cu, cuModule, "scale", blockSize, 0);
        scaleExecutionConfig = new ExecutionConfig(maxBlockParSM * smCount, 1, 1, blockSize, 1, 1, 0, null);
        maxBlockParSM = OccupancyHelper.getMaxActiveBlocksPerMultiprocessor(cu, cuModule, "memcpy", blockSize, 0);
        memcpyExecutionConfig = new ExecutionConfig(maxBlockParSM * smCount, 1, 1, blockSize, 1, 1, 0, null);
    }

    public Pointer hostRegister(Pointer hostPtr, long size) {
        cuda.cudaHostRegister(hostPtr, size, cudaHostRegisterFlag.cudaHostRegisterDefault).check();
        PointerByReference ptrRef = new PointerByReference();
        cuda.cudaHostGetDevicePointer(ptrRef, hostPtr, 0).check();
        return ptrRef.getValue();
    }

    public void hostUnregister(Pointer hostPtr) {
        cuda.cudaHostUnregister(hostPtr).check();
    }

    public void scale(Pointer data, int length, int factor) {
        kernels.scale(scaleExecutionConfig, data, length, factor).check();
        cuda.cudaDeviceSynchronize().check();
    }

    public void memcpy(Pointer dst, Pointer src, int length) {
        kernels.memcpy(memcpyExecutionConfig, dst, src, length).check();
        cuda.cudaDeviceSynchronize().check();
    }

    @Override
    public void close() {
        kernelSupport.unloadModule(cuModule);
        kernelSupport.releasePrimaryContext();
    }

}
