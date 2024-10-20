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
typedef unsigned int uint;

extern "C" __global__ void scale(long long *data, uint length, int factor)
{
    for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < length; x += gridDim.x * blockDim.x)
    {
        data[x] *= factor;
    }
}

template<typename T>
__device__ void memcpy(T *dst, T *src, uint length)
{
    for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < length; x += gridDim.x * blockDim.x)
    {
        dst[x] = src[x];
    }
}

extern "C" __global__ void memcpy_int1(int1 *dst, int1 *src, uint length)
{
    memcpy<int1>(dst, src, length);
}

extern "C" __global__ void memcpy_int2(int2 *dst, int2 *src, uint length)
{
    memcpy<int2>(dst, src, length);
}

extern "C" __global__ void memcpy_int4(int4 *dst, int4 *src, uint length)
{
    memcpy<int4>(dst, src, length);
}

extern "C" __global__ void memcpy_longlong1(longlong1 *dst, longlong1 *src, uint length)
{
    memcpy<longlong1>(dst, src, length);
}

extern "C" __global__ void memcpy_longlong2(longlong2 *dst, longlong2 *src, uint length)
{
    memcpy<longlong2>(dst, src, length);
}

extern "C" __global__ void memcpy_longlong4(longlong4 *dst, longlong4 *src, uint length)
{
    memcpy<longlong4>(dst, src, length);
}
            """; // @formatter:on

    public interface MemcpyKernels {
        /**
         * data = data * scale
         */
        CUresult scale(ExecutionConfig executionConfig, Pointer data, int length, int factor);

        CUresult memcpy_int1(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);

        CUresult memcpy_int2(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);

        CUresult memcpy_int4(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);

        CUresult memcpy_longlong1(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);

        CUresult memcpy_longlong2(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);

        CUresult memcpy_longlong4(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);
    }

    public enum MemcpyFunctionId {
        int1(0.5),
        int2(1),
        int4(2),

        longlong1(1),
        longlong2(2),
        longlong4(4);

        public final double nbLongs;

        MemcpyFunctionId(double nbLongs) {
            this.nbLongs = nbLongs;
        }
    }

    @FunctionalInterface
    interface MemcpyKernelFunction {
        CUresult memcpy(ExecutionConfig executionConfig, Pointer dst, Pointer src, int length);
    }

    public class MemcpyFunction {
        private final MemcpyFunctionId id;
        private final MemcpyKernelFunction kernelFunction;
        private final ExecutionConfig executionConfig;

        private MemcpyFunction(MemcpyFunctionId id, MemcpyKernelFunction kernelFunction, ExecutionConfig executionConfig) {
            this.id = id;
            this.executionConfig = executionConfig;
            this.kernelFunction = kernelFunction;
        }

        public void run(Pointer dst, Pointer src, int length) {
            kernelFunction.memcpy(executionConfig, src, dst, (int)(length / id.nbLongs)).check();
            cuda.cudaDeviceSynchronize().check();
        }
    }

    private final RuntimeAPI cuda;

    private final KernelSupport kernelSupport;
    private final Pointer cuModule;
    private final MemcpyKernels kernels;
    private final ExecutionConfig scaleExecutionConfig;

    private final MemcpyFunction[] memcpyFunctions = new MemcpyFunction[MemcpyFunctionId.values().length];

    public MemcpyKernelsSupport(RuntimeAPI cuda, int deviceOrdinal, int blockSize) {
        this.cuda = cuda;
        DriverAPI cu = DriverAPI.load();

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

        for (MemcpyFunctionId id : MemcpyFunctionId.values()) {
            maxBlockParSM = OccupancyHelper.getMaxActiveBlocksPerMultiprocessor(cu, cuModule, "memcpy_" + id.name(), blockSize, 0);
            ExecutionConfig executionConfig = new ExecutionConfig(maxBlockParSM * smCount, 1, 1, blockSize, 1, 1, 0, null);
            MemcpyKernelFunction kernelFunction = switch (id) {
                case int1 ->  kernels::memcpy_int1;
                case int2 ->  kernels::memcpy_int2;
                case int4 ->  kernels::memcpy_int4;
                case longlong1 ->  kernels::memcpy_longlong1;
                case longlong2 ->  kernels::memcpy_longlong2;
                case longlong4 ->  kernels::memcpy_longlong4;
            };
            memcpyFunctions[id.ordinal()] = new MemcpyFunction(id, kernelFunction, executionConfig);
        }
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

    public void memcpy(MemcpyFunctionId id, Pointer dst, Pointer src, int length) {
        MemcpyFunction memcpyFunction = memcpyFunctions[id.ordinal()];
        memcpyFunction.run(dst, src , length);
    }

    @Override
    public void close() {
        kernelSupport.unloadModule(cuModule);
        kernelSupport.releasePrimaryContext();
    }

}
