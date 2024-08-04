package og.yah.tools.cuda.cublas;

import og.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasMath_t implements NativeEnum {
    CUBLAS_DEFAULT_MATH(0),

    /* deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
    CUBLAS_TENSOR_OP_MATH(1),

    /* same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
       cudaDataType as compute type */
    CUBLAS_PEDANTIC_MATH(2),

    /* allow accelerating single precision routines using TF32 tensor cores */
    CUBLAS_TF32_TENSOR_OP_MATH(3),

    /* flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
       with lower size output type */
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION(16);

    private final int nativeValue;

    cublasMath_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }


}
