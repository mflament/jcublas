package og.yah.tools.cuda.cublas;

import og.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cudaDataType_t implements NativeEnum {
    CUDA_R_16F( 2), /* real as a half */
    CUDA_C_16F( 6), /* complex as a pair of half numbers */
    CUDA_R_16BF(14), /* real as a nv_bfloat16 */
    CUDA_C_16BF(15), /* complex as a pair of nv_bfloat16 numbers */
    CUDA_R_32F( 0), /* real as a float */
    CUDA_C_32F( 4), /* complex as a pair of float numbers */
    CUDA_R_64F( 1), /* real as a double */
    CUDA_C_64F( 5), /* complex as a pair of double numbers */
    CUDA_R_4I(16), /* real as a signed 4-bit int */
    CUDA_C_4I(17), /* complex as a pair of signed 4-bit int numbers */
    CUDA_R_4U(18), /* real as a unsigned 4-bit int */
    CUDA_C_4U(19), /* complex as a pair of unsigned 4-bit int numbers */
    CUDA_R_8I( 3), /* real as a signed 8-bit int */
    CUDA_C_8I( 7), /* complex as a pair of signed 8-bit int numbers */
    CUDA_R_8U( 8), /* real as a unsigned 8-bit int */
    CUDA_C_8U( 9), /* complex as a pair of unsigned 8-bit int numbers */
    CUDA_R_16I(20), /* real as a signed 16-bit int */
    CUDA_C_16I(21), /* complex as a pair of signed 16-bit int numbers */
    CUDA_R_16U(22), /* real as a unsigned 16-bit int */
    CUDA_C_16U(23), /* complex as a pair of unsigned 16-bit int numbers */
    CUDA_R_32I(10), /* real as a signed 32-bit int */
    CUDA_C_32I(11), /* complex as a pair of signed 32-bit int numbers */
    CUDA_R_32U(12), /* real as a unsigned 32-bit int */
    CUDA_C_32U(13), /* complex as a pair of unsigned 32-bit int numbers */
    CUDA_R_64I(24), /* real as a signed 64-bit int */
    CUDA_C_64I(25), /* complex as a pair of signed 64-bit int numbers */
    CUDA_R_64U(26), /* real as a unsigned 64-bit int */
    CUDA_C_64U(27), /* complex as a pair of unsigned 64-bit int numbers */
    CUDA_R_8F_E4M3(28), /* real as a nv_fp8_e4m3 */
    CUDA_R_8F_E5M2(29); /* real as a nv_fp8_e5m2 */

    private final int nativeValue;

    cudaDataType_t(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
