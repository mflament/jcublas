package org.yah.tools.cuda.nvrtc;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.jna.CudaLibrarySupport;

/**
 * <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc">CUDA NVRTC</a>
 *
 * @version 12.4.127
 */
@SuppressWarnings("unused")
public interface NVRTC extends Library {
    /**
     * @param result <i>nvrtcResult</i>
     * @return const char *
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv419nvrtcGetErrorString11nvrtcResult">nvrtcGetErrorString</a>
     */
    Pointer nvrtcGetErrorString(nvrtcResult result);

    /**
     * @param prog <i>Pointer *</i>
     * @param src <i>const char *</i>
     * @param name <i>const char *</i>
     * @param numHeaders <i>int</i>
     * @param headers <i>const char * const *</i>
     * @param includeNames <i>const char * const *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv418nvrtcCreateProgramP12nvrtcProgramPKcPKciPPCKcPPCKc">nvrtcCreateProgram</a>
     */
    nvrtcResult nvrtcCreateProgram(PointerByReference prog, String src, String name, int numHeaders, String[] headers, String[] includeNames);

    /**
     * @param prog <i>Pointer *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv419nvrtcDestroyProgramP12nvrtcProgram">nvrtcDestroyProgram</a>
     */
    nvrtcResult nvrtcDestroyProgram(PointerByReference prog);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param numOptions <i>int</i>
     * @param options <i>const char * const *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv419nvrtcCompileProgram12nvrtcProgramiPPCKc">nvrtcCompileProgram</a>
     */
    nvrtcResult nvrtcCompileProgram(Pointer prog, int numOptions, String[] options);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param ptxSizeRet <i>size_t *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv415nvrtcGetPTXSize12nvrtcProgramP6size_t">nvrtcGetPTXSize</a>
     */
    nvrtcResult nvrtcGetPTXSize(Pointer prog, LongByReference ptxSizeRet);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param ptx <i>char *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv411nvrtcGetPTX12nvrtcProgramPc">nvrtcGetPTX</a>
     */
    nvrtcResult nvrtcGetPTX(Pointer prog, Pointer ptx);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param cubinSizeRet <i>size_t *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv417nvrtcGetCUBINSize12nvrtcProgramP6size_t">nvrtcGetCUBINSize</a>
     */
    nvrtcResult nvrtcGetCUBINSize(Pointer prog, LongByReference cubinSizeRet);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param cubin <i>char *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv413nvrtcGetCUBIN12nvrtcProgramPc">nvrtcGetCUBIN</a>
     */
    nvrtcResult nvrtcGetCUBIN(Pointer prog, Pointer cubin);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param logSizeRet <i>size_t *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv422nvrtcGetProgramLogSize12nvrtcProgramP6size_t">nvrtcGetProgramLogSize</a>
     */
    nvrtcResult nvrtcGetProgramLogSize(Pointer prog, LongByReference logSizeRet);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param log <i>char *</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv418nvrtcGetProgramLog12nvrtcProgramPc">nvrtcGetProgramLog</a>
     */
    nvrtcResult nvrtcGetProgramLog(Pointer prog, Pointer log);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param name_expression <i>const char * const</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv422nvrtcAddNameExpression12nvrtcProgramPCKc">nvrtcAddNameExpression</a>
     */
    nvrtcResult nvrtcAddNameExpression(Pointer prog, String name_expression);

    /**
     * @param prog <i>nvrtcProgram</i>
     * @param name_expression <i>const char *const</i>
     * @param lowered_name <i>const char**</i>
     * @return {@link nvrtcResult}
     * @see <a href="https://docs.nvidia.com/cuda/archive/12.4.1/nvrtc/index.html#_CPPv419nvrtcGetLoweredName12nvrtcProgramPCKcPPKc">nvrtcGetLoweredName</a>
     */
    nvrtcResult nvrtcGetLoweredName(Pointer prog, String name_expression, PointerByReference lowered_name);

    static NVRTC load() {
        return load(CudaLibrarySupport.lookupLibrary("nvrtc64"));
    }

    static NVRTC load(String name) {
        return Native.load(name, NVRTC.class);
    }

}
