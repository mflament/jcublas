package org.yah.tools.cuda.driver;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;

import javax.annotation.Nullable;

@SuppressWarnings("unused")
public interface DriverAPI extends Library {

    /**
     * Gets the string description of an error code
     *
     * @param error <i>CUresult</i>
     * @param pStr  <i>const char **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada">cuGetErrorString</a>
     */
    CUresult cuGetErrorString(CUresult error, PointerByReference pStr);

    /**
     * Gets the string representation of an error code enum name
     *
     * @param error <i>CUresult</i>
     * @param pStr  <i>const char **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468">cuGetErrorName</a>
     */
    CUresult cuGetErrorName(CUresult error, PointerByReference pStr);

    /**
     * Initialize the CUDA driver API
     * Initializes the driver API and must be called before any other function from
     * the driver API in the current process. Currently, the \p Flags parameter must be 0. If ::cuInit()
     * has not been called, any function from the driver API will return
     * ::CUDA_ERROR_NOT_INITIALIZED.
     *
     * @param Flags <i>unsigned int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3">cuInit</a>
     */
    CUresult cuInit(int Flags);

    /**
     * Returns the latest CUDA version supported by driver
     *
     * @param driverVersion <i>int *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71">cuDriverGetVersion</a>
     */
    CUresult cuDriverGetVersion(Pointer driverVersion);

    /**
     * Returns a handle to a compute device
     *
     * @param device  <i>CUdevice *</i>
     * @param ordinal <i>int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb">cuDeviceGet</a>
     */
    CUresult cuDeviceGet(PointerByReference device, int ordinal);

    /**
     * Returns the number of compute-capable devices
     *
     * @param count <i>int *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74">cuDeviceGetCount</a>
     */
    CUresult cuDeviceGetCount(IntByReference count);

    /**
     * Returns an identifier string for the device
     *
     * @param name <i>char *</i>
     * @param len  <i>int</i>
     * @param dev  <i>CUdevice</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f">cuDeviceGetName</a>
     */
    CUresult cuDeviceGetName(Pointer name, int len, Pointer dev);

    /**
     * Returns information about the device
     *
     * @param pi     <i>int *</i>
     * @param attrib <i>CUdevice_attribute</i>
     * @param dev    <i>CUdevice</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266">cuDeviceGetAttribute</a>
     */
    CUresult cuDeviceGetAttribute(IntByReference pi, CUdevice_attribute attrib, Pointer dev);

    /**
     * Retain the primary context on the GPU
     *
     * @param pctx <i>CUcontext *</i>
     * @param dev  <i>CUdevice</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300">cuDevicePrimaryCtxRetain</a>
     */
    CUresult cuDevicePrimaryCtxRetain(PointerByReference pctx, Pointer dev);

    /**
     * Release the primary context on the GPU
     *
     * @param dev <i>CUdevice</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad">cuDevicePrimaryCtxRelease</a>
     */
    CUresult cuDevicePrimaryCtxRelease(Pointer dev);

    /**
     * Set flags for the primary context
     *
     * @param dev   <i>CUdevice</i>
     * @param flags <i>unsigned int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf">cuDevicePrimaryCtxSetFlags</a>
     */
    CUresult cuDevicePrimaryCtxSetFlags(Pointer dev, int flags);

    /**
     * Get the state of the primary context
     *
     * @param dev    <i>CUdevice</i>
     * @param flags  <i>unsigned int *</i>
     * @param active <i>int *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469">cuDevicePrimaryCtxGetState</a>
     */
    CUresult cuDevicePrimaryCtxGetState(Pointer dev, Pointer flags, Pointer active);

    /**
     * Destroy all allocations and reset all state on the primary context
     *
     * @param dev <i>CUdevice</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12">cuDevicePrimaryCtxReset</a>
     */
    CUresult cuDevicePrimaryCtxReset(Pointer dev);

    /**
     * Returns the device ID for the current context
     *
     * @param device <i>CUdevice *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e">cuCtxGetDevice</a>
     */
    CUresult cuCtxGetDevice(PointerByReference device);

    /**
     * Block for a context's tasks to complete
     *
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616">cuCtxSynchronize</a>
     */
    CUresult cuCtxSynchronize();

    /**
     * Loads a compute module
     *
     * @param module <i>CUmodule *</i>
     * @param fname  <i>const char *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3">cuModuleLoad</a>
     */
    CUresult cuModuleLoad(PointerByReference module, Pointer fname);

    /**
     * Load a module's data
     *
     * @param module <i>CUmodule *</i>
     * @param image  <i>const void *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b">cuModuleLoadData</a>
     */
    CUresult cuModuleLoadData(PointerByReference module, Pointer image);

    /**
     * Load a module's data with options
     *
     * @param module       <i>CUmodule *</i>
     * @param image        <i>const void *</i>
     * @param numOptions   <i>unsigned int</i>
     * @param options      <i>CUjit_option *</i>
     * @param optionValues <i>void **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12">cuModuleLoadDataEx</a>
     */
    CUresult cuModuleLoadDataEx(PointerByReference module, Pointer image, int numOptions, Pointer options, PointerByReference optionValues);

    /**
     * Load a module's data
     *
     * @param module   <i>CUmodule *</i>
     * @param fatCubin <i>const void *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b">cuModuleLoadFatBinary</a>
     */
    CUresult cuModuleLoadFatBinary(PointerByReference module, Pointer fatCubin);

    /**
     * Unloads a module
     *
     * @param hmod <i>CUmodule</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b">cuModuleUnload</a>
     */
    CUresult cuModuleUnload(Pointer hmod);

    /**
     * Returns a function handle
     *
     * @param hfunc <i>CUfunction *</i>
     * @param hmod  <i>CUmodule</i>
     * @param name  <i>const char *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf">cuModuleGetFunction</a>
     */
    CUresult cuModuleGetFunction(PointerByReference hfunc, Pointer hmod, String name);

    /**
     * Returns the number of functions within a module
     *
     * @param count <i>unsigned int *</i>
     * @param mod   <i>CUmodule</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gecc8fb61eca765cb0f1eb32f00cf3b49">cuModuleGetFunctionCount</a>
     */
    CUresult cuModuleGetFunctionCount(IntByReference count, Pointer mod);

    /**
     * Returns the function handles within a module.
     *
     * @param functions    <i>CUfunction *</i>
     * @param numFunctions <i>unsigned int</i>
     * @param mod          <i>CUmodule</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50">cuModuleEnumerateFunctions</a>
     */
    CUresult cuModuleEnumerateFunctions(Pointer functions, int numFunctions, Pointer mod);

    /**
     * Returns a global pointer from a module
     *
     * @param dptr  <i>CUdeviceptr *</i>
     * @param bytes <i>size_t *</i>
     * @param hmod  <i>CUmodule</i>
     * @param name  <i>const char *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab">cuModuleGetGlobal</a>
     */
    CUresult cuModuleGetGlobal(Pointer dptr, long bytes, Pointer hmod, Pointer name);

    /**
     * Creates a pending JIT linker invocation.
     *
     * @param numOptions   <i>unsigned int</i>
     * @param options      <i>CUjit_option *</i>
     * @param optionValues <i>void **</i>
     * @param stateOut     <i>CUlinkState *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d">cuLinkCreate</a>
     */
    CUresult cuLinkCreate(int numOptions, Pointer options, PointerByReference optionValues, PointerByReference stateOut);

    /**
     * Add an input to a pending linker invocation
     *
     * @param state        <i>CUlinkState</i>
     * @param type         <i>CUjitInputType</i>
     * @param data         <i>void *</i>
     * @param size         <i>size_t</i>
     * @param name         <i>const char *</i>
     * @param numOptions   <i>unsigned int</i>
     * @param options      <i>CUjit_option *</i>
     * @param optionValues <i>void **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77">cuLinkAddData</a>
     */
    CUresult cuLinkAddData(Pointer state, CUjitInputType type, Pointer data, long size, Pointer name, int numOptions, Pointer options, PointerByReference optionValues);

    /**
     * Add a file input to a pending linker invocation
     *
     * @param state        <i>CUlinkState</i>
     * @param type         <i>CUjitInputType</i>
     * @param path         <i>const char *</i>
     * @param numOptions   <i>unsigned int</i>
     * @param options      <i>CUjit_option *</i>
     * @param optionValues <i>void **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c">cuLinkAddFile</a>
     */
    CUresult cuLinkAddFile(Pointer state, CUjitInputType type, Pointer path, int numOptions, Pointer options, PointerByReference optionValues);

    /**
     * Complete a pending linker invocation
     *
     * @param state    <i>CUlinkState</i>
     * @param cubinOut <i>void **</i>
     * @param sizeOut  <i>size_t *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716">cuLinkComplete</a>
     */
    CUresult cuLinkComplete(Pointer state, PointerByReference cubinOut, long sizeOut);

    /**
     * Destroys state for a JIT linker invocation.
     *
     * @param state <i>CUlinkState</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9">cuLinkDestroy</a>
     */
    CUresult cuLinkDestroy(Pointer state);

    /**
     * Load a library with specified code and options
     *
     * @param library             <i>CUlibrary *</i>
     * @param code                <i>const void *</i>
     * @param jitOptions          <i>CUjit_option *</i>
     * @param jitOptionsValues    <i>void **</i>
     * @param numJitOptions       <i>unsigned int</i>
     * @param libraryOptions      <i>CUlibraryOption *</i>
     * @param libraryOptionValues <i>void**</i>
     * @param numLibraryOptions   <i>unsigned int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d">cuLibraryLoadData</a>
     */
    CUresult cuLibraryLoadData(PointerByReference library, Pointer code, Pointer jitOptions, PointerByReference jitOptionsValues, int numJitOptions, Pointer libraryOptions, PointerByReference libraryOptionValues, int numLibraryOptions);

    /**
     * Load a library with specified file and options
     *
     * @param library             <i>CUlibrary *</i>
     * @param fileName            <i>const char *</i>
     * @param jitOptions          <i>CUjit_option *</i>
     * @param jitOptionsValues    <i>void **</i>
     * @param numJitOptions       <i>unsigned int</i>
     * @param libraryOptions      <i>CUlibraryOption *</i>
     * @param libraryOptionValues <i>void **</i>
     * @param numLibraryOptions   <i>unsigned int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205">cuLibraryLoadFromFile</a>
     */
    CUresult cuLibraryLoadFromFile(PointerByReference library, Pointer fileName, Pointer jitOptions, PointerByReference jitOptionsValues, int numJitOptions, Pointer libraryOptions, PointerByReference libraryOptionValues, int numLibraryOptions);

    /**
     * Unloads a library
     *
     * @param library <i>CUlibrary</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914">cuLibraryUnload</a>
     */
    CUresult cuLibraryUnload(Pointer library);

    /**
     * Returns a kernel handle
     *
     * @param pKernel <i>CUkernel *</i>
     * @param library <i>CUlibrary</i>
     * @param name    <i>const char *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a">cuLibraryGetKernel</a>
     */
    CUresult cuLibraryGetKernel(PointerByReference pKernel, Pointer library, Pointer name);

    /**
     * Returns the number of kernels within a library
     *
     * @param count <i>unsigned int *</i>
     * @param lib   <i>CUlibrary</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g142732b1c9afaa662f21cae9a558d2d4">cuLibraryGetKernelCount</a>
     */
    CUresult cuLibraryGetKernelCount(Pointer count, Pointer lib);

    /**
     * Retrieve the kernel handles within a library.
     *
     * @param kernels    <i>CUkernel *</i>
     * @param numKernels <i>unsigned int</i>
     * @param lib        <i>CUlibrary</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ga8ae2f42ab3a8fe789ac2dced8219608">cuLibraryEnumerateKernels</a>
     */
    CUresult cuLibraryEnumerateKernels(PointerByReference kernels, int numKernels, Pointer lib);

    /**
     * Returns a module handle
     *
     * @param pMod    <i>CUmodule *</i>
     * @param library <i>CUlibrary</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8">cuLibraryGetModule</a>
     */
    CUresult cuLibraryGetModule(PointerByReference pMod, Pointer library);

    /**
     * Returns a function handle
     *
     * @param pFunc  <i>CUfunction *</i>
     * @param kernel <i>CUkernel</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a">cuKernelGetFunction</a>
     */
    CUresult cuKernelGetFunction(PointerByReference pFunc, Pointer kernel);

    /**
     * Returns the function name for a ::CUkernel handle
     *
     * @param name  <i>const char **</i>
     * @param hfunc <i>CUkernel</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge758151073b777ef3ba11a45f7d22adf">cuKernelGetName</a>
     */
    CUresult cuKernelGetName(PointerByReference name, Pointer hfunc);

    /**
     * Returns the offset and size of a kernel parameter in the device-side parameter layout
     *
     * @param kernel      <i>CUkernel</i>
     * @param paramIndex  <i>size_t</i>
     * @param paramOffset <i>size_t *</i>
     * @param paramSize   <i>size_t *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ga61653c9f13f713527e189fb0c2fe235">cuKernelGetParamInfo</a>
     */
    CUresult cuKernelGetParamInfo(Pointer kernel, long paramIndex, LongByReference paramOffset, LongByReference paramSize);

    /**
     * Returns the function name for a ::CUfunction handle
     *
     * @param name  <i>const char **</i>
     * @param hfunc <i>CUfunction</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gf60c6c51203cab164c07d6ddcc2b2e26">cuFuncGetName</a>
     */
    CUresult cuFuncGetName(PointerByReference name, Pointer hfunc);

    /**
     * Returns the offset and size of a kernel parameter in the device-side parameter layout
     *
     * @param func        <i>CUfunction</i>
     * @param paramIndex  <i>size_t</i>
     * @param paramOffset <i>size_t *</i>
     * @param paramSize   <i>size_t *</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g6874b82bcf2803902085645e46e0ca0e">cuFuncGetParamInfo</a>
     */
    CUresult cuFuncGetParamInfo(Pointer func, long paramIndex, LongByReference paramOffset, LongByReference paramSize);

    /**
     * Launches a CUDA function ::CUfunction or a CUDA kernel ::CUkernel
     *
     * @param f              <i>CUfunction</i>
     * @param gridDimX       <i>unsigned int</i>
     * @param gridDimY       <i>unsigned int</i>
     * @param gridDimZ       <i>unsigned int</i>
     * @param blockDimX      <i>unsigned int</i>
     * @param blockDimY      <i>unsigned int</i>
     * @param blockDimZ      <i>unsigned int</i>
     * @param sharedMemBytes <i>unsigned int</i>
     * @param hStream        <i>CUstream</i>
     * @param kernelParams   <i>void **</i>
     * @param extra          <i>void **</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15">cuLaunchKernel</a>
     */
    CUresult cuLaunchKernel(Pointer f, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, Pointer hStream, Pointer[] kernelParams, Pointer[] extra);


    // 6.25. Occupancy https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03

    /**
     * Returns occupancy of a function
     *
     * @param numBlocks       <i>int *</i>
     * @param func            <i>CUfunction</i>
     * @param blockSize       <i>int</i>
     * @param dynamicSMemSize <i>size_t</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98">cuOccupancyMaxActiveBlocksPerMultiprocessor</a>
     */
    CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(IntByReference numBlocks, Pointer func, int blockSize, Pointer dynamicSMemSize);

    /**
     * Suggest a launch configuration with reasonable occupancy
     *
     * @param minGridSize <i>int *</i>
     * @param blockSize <i>int *</i>
     * @param func <i>CUfunction</i>
     * @param blockSizeToDynamicSMemSize <i>CUoccupancyB2DSize</i>
     * @param dynamicSMemSize <i>size_t</i>
     * @param blockSizeLimit <i>int</i>
     * @return {@link CUresult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03">cuOccupancyMaxPotentialBlockSize</a>
     */
    CUresult cuOccupancyMaxPotentialBlockSize(IntByReference minGridSize, IntByReference blockSize, Pointer func, @Nullable CUoccupancyB2DSize blockSizeToDynamicSMemSize, Pointer dynamicSMemSize, int blockSizeLimit);

    static DriverAPI load() {
        return load("nvcuda");
    }

    static DriverAPI load(String name) {
        DriverAPI driverAPI = Native.load(name, DriverAPI.class);
        driverAPI.cuInit(0).check();
        return driverAPI;
    }

}
