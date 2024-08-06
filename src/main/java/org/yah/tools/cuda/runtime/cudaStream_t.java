package org.yah.tools.cuda.runtime;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

@SuppressWarnings("unused")
public class cudaStream_t extends Pointer {

    public cudaStream_t(long peer) {
        super(peer);
    }

    public cudaStream_t(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public cudaStream_t getValue() {
            return new cudaStream_t(super.getValue());
        }
    }

}
