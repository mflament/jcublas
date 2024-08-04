package og.yah.tools.cuda.cublas;

import og.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasPointerMode_t implements NativeEnum {
    CUBLAS_POINTER_MODE_HOST(0),
    CUBLAS_POINTER_MODE_DEVICE( 1);

    private final int nativeValue;

    cublasPointerMode_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
