package org.yah.tools.cuda.cublas;

import org.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasAtomicsMode_t implements NativeEnum {
    CUBLAS_ATOMICS_NOT_ALLOWED(0),
    CUBLAS_ATOMICS_ALLOWED( 1);

    private final int nativeValue;

    cublasAtomicsMode_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
