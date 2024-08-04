package og.yah.tools.cuda.cublas;

import og.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cublasComputeType_t implements NativeEnum {
    CUBLAS_COMPUTE_16F(64),           /* half - default */
    CUBLAS_COMPUTE_16F_PEDANTIC(65),  /* half - pedantic */
    CUBLAS_COMPUTE_32F(68),           /* float - default */
    CUBLAS_COMPUTE_32F_PEDANTIC(69),  /* float - pedantic */
    CUBLAS_COMPUTE_32F_FAST_16F(74),  /* float - fast, allows down-converting inputs to half or TF32 */
    CUBLAS_COMPUTE_32F_FAST_16BF(75), /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
    CUBLAS_COMPUTE_32F_FAST_TF32(77), /* float - fast, allows down-converting inputs to TF32 */
    CUBLAS_COMPUTE_64F(70),           /* double - default */
    CUBLAS_COMPUTE_64F_PEDANTIC(71),  /* double - pedantic */
    CUBLAS_COMPUTE_32I(72),           /* signed 32-bit int - default */
    CUBLAS_COMPUTE_32I_PEDANTIC(73);  /* signed 32-bit int - pedantic */

    private final int nativeValue;

    cublasComputeType_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
