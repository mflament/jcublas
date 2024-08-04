package og.yah.tools.cuda.cublas;

import com.sun.jna.FunctionMapper;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.FloatByReference;
import com.sun.jna.ptr.IntByReference;
import og.yah.tools.cuda.jna.CudaLibrarySupport;
import og.yah.tools.cuda.jna.NativeEnumPointer;
import og.yah.tools.cuda.jna.size_t;
import og.yah.tools.cuda.runtime.cudaStream_t;

import javax.annotation.Nullable;
import java.util.Map;

@SuppressWarnings("unused")
public interface CublasAPI extends Library {

    cublasStatus_t cublasCreate(cublasHandle_t.ByReference handle);

    cublasStatus_t cublasDestroy(cublasHandle_t handle);

    cublasStatus_t cublasGetVersion(cublasHandle_t handle, Pointer version);

    cublasStatus_t cublasGetVersion(@Nullable cublasHandle_t handle, IntByReference version);

    cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, Pointer workspace, size_t workspaceSizeInBytes);

    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);

    cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t.ByReference streamId);

    cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, NativeEnumPointer<cublasPointerMode_t> mode);

    cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);

    cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, NativeEnumPointer<cublasAtomicsMode_t> mode);

    cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);

    cublasStatus_t cublasGetMathMode(cublasHandle_t handle, NativeEnumPointer<cublasMath_t> mode);

    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode);

    cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, Pointer smCountTarget);

    cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget);

    String cublasGetStatusName(cublasStatus_t status);

    String cublasGetStatusString(cublasStatus_t status);

    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               Pointer alpha,
                               Pointer A, int lda,
                               Pointer B, int ldb,
                               Pointer beta,
                               Pointer C, int ldc);

    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               FloatByReference alpha,
                               Pointer A, int lda,
                               Pointer B, int ldb,
                               FloatByReference beta,
                               Pointer C, int ldc);

    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               Pointer alpha,
                               Pointer A, int lda,
                               Pointer B, int ldb,
                               Pointer beta,
                               Pointer C, int ldc);

    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               DoubleByReference alpha,
                               Pointer A, int lda,
                               Pointer B, int ldb,
                               DoubleByReference beta,
                               Pointer C, int ldc);

    cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m, int n, int k,
                                Pointer alpha,
                                Pointer A,
                                cudaDataType_t Atype,
                                int lda,
                                Pointer B,
                                cudaDataType_t Btype,
                                int ldb,
                                Pointer beta,
                                Pointer C,
                                cudaDataType_t Ctype,
                                int ldc,
                                cublasComputeType_t computeType,
                                cublasGemmAlgo_t algo);

    static CublasAPI load() {
        return load(CudaLibrarySupport.lookupLibrary("cublas"));
    }

    static CublasAPI load(String name) {
        FunctionMapper functionMapper = (library, method) -> {
            String javaName = method.getName();
            if (javaName.equals("cublasSetAtomicsMode") || javaName.equals("cublasSetMathMode") || javaName.equals("cublasGemmEx"))
                return javaName;
            if (javaName.endsWith("_64"))
                return javaName.substring(0, javaName.length() - 3) + "_v2_64";
            return javaName + "_v2";
        };
        return Native.load(name, CublasAPI.class, Map.of(Library.OPTION_FUNCTION_MAPPER, functionMapper));
    }
}
