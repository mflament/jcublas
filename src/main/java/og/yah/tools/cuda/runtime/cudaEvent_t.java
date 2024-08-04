package og.yah.tools.cuda.runtime;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

@SuppressWarnings("unused")
public class cudaEvent_t extends Pointer {

    public cudaEvent_t(long peer) {
        super(peer);
    }

    public cudaEvent_t(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public cudaEvent_t getValue() {
            return new cudaEvent_t(super.getValue());
        }
    }

}
