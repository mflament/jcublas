package org.yah.tools.cuda.nvrtc;

import org.yah.tools.cuda.jna.NativeEnum;

/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 * NVRTC API functions return nvrtcResult to indicate the call
 * result.
 */
@SuppressWarnings("unused")
public enum nvrtcResult implements NativeEnum {

    NVRTC_SUCCESS(0),
    NVRTC_ERROR_OUT_OF_MEMORY(1),
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE(2),
    NVRTC_ERROR_INVALID_INPUT(3),
    NVRTC_ERROR_INVALID_PROGRAM(4),
    NVRTC_ERROR_INVALID_OPTION(5),
    NVRTC_ERROR_COMPILATION(6),
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE(7),
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION(8),
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION(9),
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID(10),
    NVRTC_ERROR_INTERNAL_ERROR(11),
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED(12);

    private final int value;

    nvrtcResult(int value) {
        this.value = value;
    }

    @Override
    public int nativeValue() {
        return value;
    }

    public void check() {
        if (this != nvrtcResult.NVRTC_SUCCESS)
            throw new NVRTCException(this);
    }
}
