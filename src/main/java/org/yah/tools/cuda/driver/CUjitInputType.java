package org.yah.tools.cuda.driver;

import org.yah.tools.cuda.jna.NativeEnum;

/**
 * Device code formats
 */
@SuppressWarnings("unused")
public enum CUjitInputType implements NativeEnum {

    /**
     * Compiled device-class-specific device code\n
     * Applicable options: none
     */
    CU_JIT_INPUT_CUBIN(0),
    /**
     * PTX source code\n
     * Applicable options: PTX compiler options
     */
    CU_JIT_INPUT_PTX(1),
    /**
     * Bundle of multiple cubins and/or PTX of some device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    CU_JIT_INPUT_FATBINARY(2),
    /**
     * Host object with embedded device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    CU_JIT_INPUT_OBJECT(3),
    /**
     * Archive of host objects with embedded device code\n
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    CU_JIT_INPUT_LIBRARY(4),
    /**
     * @deprecated High-level intermediate code for link-time optimization\n
     * Applicable options: NVVM compiler options, PTX compiler options
     *      <p>
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    @Deprecated
    CU_JIT_INPUT_NVVM(5),
    CU_JIT_NUM_INPUT_TYPES(6);

    private final int value;

    CUjitInputType(int value) {
        this.value = value;
    }

    @Override
    public int nativeValue() {
        return value;
    }
}
