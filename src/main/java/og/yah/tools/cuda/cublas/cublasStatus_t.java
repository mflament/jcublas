package og.yah.tools.cuda.cublas;

import og.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasStatus_t implements NativeEnum {

    CUBLAS_STATUS_SUCCESS(0),
    CUBLAS_STATUS_NOT_INITIALIZED(1),
    CUBLAS_STATUS_ALLOC_FAILED(3),
    CUBLAS_STATUS_INVALID_VALUE(7),
    CUBLAS_STATUS_ARCH_MISMATCH(8),
    CUBLAS_STATUS_MAPPING_ERROR(11),
    CUBLAS_STATUS_EXECUTION_FAILED(13),
    CUBLAS_STATUS_INTERNAL_ERROR(14),
    CUBLAS_STATUS_NOT_SUPPORTED(15),
    CUBLAS_STATUS_LICENSE_ERROR(16);

    private final int nativeValue;

    cublasStatus_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }

    public void check() {
        if (this != CUBLAS_STATUS_SUCCESS)
            throw new CublasException(this);
    }
}
