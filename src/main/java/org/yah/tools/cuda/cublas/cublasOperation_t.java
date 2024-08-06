package org.yah.tools.cuda.cublas;

import org.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasOperation_t implements NativeEnum {
    CUBLAS_OP_N(0),
    CUBLAS_OP_T(1),
    CUBLAS_OP_C(2),
    /* synonym if CUBLAS_OP_C */
    CUBLAS_OP_HERMITAN(2),
    /* conjugate, placeholder - not supported in the current release */
    CUBLAS_OP_CONJG(3);

    private final int nativeValue;

    cublasOperation_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
