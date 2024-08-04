package og.yah.tools.cuda.jna;

import com.sun.jna.FromNativeContext;
import com.sun.jna.Native;
import com.sun.jna.NativeMapped;

public class size_t implements NativeMapped {

    private static final int SIZE = checkSize();

    private static int checkSize() {
        int size = Native.SIZE_T_SIZE;
        if (size == 4 || size == 8)
            return size;
        throw new IllegalStateException("Unsupported size_t size " + size);
    }

    private long value;

    public size_t() {
        this(0);
    }

    public size_t(long value) {
        this.value = value;
    }

    public long getValue() {
        return value;
    }

    public void setValue(long value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return Long.toUnsignedString(value);
    }

    @Override
    public Object fromNative(Object nativeValue, FromNativeContext context) {
        if (SIZE == 4)
            this.value = (int) nativeValue;
        else
            this.value = (long) nativeValue;
        return this;
    }

    @Override
    public Object toNative() {
        return SIZE == 4 ? (int) value : value;
    }

    @Override
    public Class<?> nativeType() {
        return SIZE == 4 ? int.class : long.class;
    }

    @SuppressWarnings("unused")
    public static class ByReference extends com.sun.jna.ptr.ByReference {
        public ByReference() {
            super(SIZE);
        }

        public size_t getValue() {
            long value;
            if (SIZE == 4)
                value = getPointer().getInt(0);
            else
                value = getPointer().getLong(0);
            return new size_t(value);
        }

        public void setValue(size_t value) {
            if (SIZE == 4)
                getPointer().setInt(0, (int) value.getValue());
            getPointer().setLong(0, value.getValue());
        }

    }
}
