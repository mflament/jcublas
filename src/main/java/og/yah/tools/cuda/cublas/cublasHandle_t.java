package og.yah.tools.cuda.cublas;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

@SuppressWarnings("unused")
public class cublasHandle_t extends Pointer {

    public cublasHandle_t(long peer) {
        super(peer);
    }

    public cublasHandle_t(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public cublasHandle_t getValue() {
            return new cublasHandle_t(super.getValue());
        }
    }

}
