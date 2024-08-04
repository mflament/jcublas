package og.yah.tools.cuda.runtime;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import og.yah.tools.cuda.jna.CudaLibrarySupport;
import og.yah.tools.cuda.jna.size_t;

import javax.annotation.Nullable;
import java.io.PipedOutputStream;

@SuppressWarnings("unused")
public interface RuntimeAPI extends Library {

    cudaError_t cudaRuntimeGetVersion(IntByReference runtimeVersion);

    cudaError_t cudaDriverGetVersion(IntByReference driverVersion);

    cudaError_t cudaGetDevice(Pointer device);

    cudaError_t cudaSetDevice(int device);

    cudaError_t cudaGetDeviceCount(Pointer count);

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp.ByReference prop, int device);

    cudaError_t cudaDeviceReset();

    // device management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE
    cudaError_t cudaDeviceSynchronize();

    cudaError_t cudaGetLastError();

    Pointer cudaGetErrorString(cudaError_t error);

    // memory management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
    cudaError_t cudaMalloc(PointerByReference devPtr, size_t size);

    cudaError_t cudaFree(Pointer devPtr);

    cudaError_t cudaMemcpy(Pointer dst, Pointer src, size_t count, cudaMemcpyKind kind);

    cudaError_t cudaMemset(Pointer devPtr, int value, size_t count);

    cudaError_t cudaMemGetInfo(size_t.ByReference free, size_t.ByReference total);

    cudaError_t cudaHostRegister(PointerByReference ptr, size_t size, int flags);

    cudaError_t cudaHostUnregister(Pointer ptr);

    cudaError_t cudaMallocHost(PointerByReference ptr, size_t size);

    cudaError_t cudaHostGetDevicePointer(PointerByReference pDevice, Pointer pHost, int flags);

    cudaError_t cudaLaunchCooperativeKernel(Pointer func, dim3 gridDim, dim3 blockDim, Pointer[] args, size_t sharedMem, cudaStream_t stream);

    cudaError_t cudaLaunchKernel(Pointer func, dim3 gridDim, dim3 blockDim, Pointer[] args, size_t sharedMem, cudaStream_t stream);

    cudaError_t cudaStreamCreate(cudaStream_t.ByReference pStream);

    cudaError_t cudaStreamCreateWithFlags(cudaStream_t.ByReference pStream, int flags);

    cudaError_t cudaStreamCreateWithPriority(cudaStream_t.ByReference pStream, int flags, int priority);

    cudaError_t cudaStreamDestroy(cudaStream_t stream);

    cudaError_t cudaStreamGetId(cudaStream_t hStream, LongByReference streamId);

    cudaError_t cudaMallocAsync(PointerByReference devPtr, size_t size, cudaStream_t hStream);

    cudaError_t cudaFreeAsync(PipedOutputStream devPtr, cudaStream_t hStream);

    cudaError_t cudaMemcpyAsync(Pointer dst, Pointer src, size_t count, cudaMemcpyKind kind, @Nullable cudaStream_t stream);

    cudaError_t cudaMemsetAsync(Pointer devPtr, int value, size_t count, @Nullable cudaStream_t stream);

    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, int flags);

    @Deprecated
    cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, Pointer userData, int flags);

    cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, Pointer userData);

    /**
     * @param stream the stream to query
     * @return cudaSuccess if all operations in stream have completed, or cudaErrorNotReady if not.
     */
    cudaError_t cudaStreamQuery(cudaStream_t stream);

    cudaError_t cudaStreamSynchronize(cudaStream_t stream);

    cudaError_t cudaGetFuncBySymbol(PointerByReference functionPtr, Pointer symbolPtr);

    static RuntimeAPI load() {
        return load(CudaLibrarySupport.lookupLibrary("cudart"));
    }

    static RuntimeAPI load(String name) {
        return Native.load(name, RuntimeAPI.class);
    }

}
