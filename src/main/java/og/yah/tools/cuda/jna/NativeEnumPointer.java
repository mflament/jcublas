package og.yah.tools.cuda.jna;

import com.sun.jna.Memory;

import java.util.Objects;

@SuppressWarnings("unused")
public class NativeEnumPointer<E extends Enum<E> & NativeEnum> extends Memory {

    private final Class<E> enumClass;

    public NativeEnumPointer(Class<E> enumClass) {
        super(Integer.BYTES);
        this.enumClass = Objects.requireNonNull(enumClass, "enumClass is null");
    }

    public Class<E> enumClass() {
        return enumClass;
    }

    public E value() {
        return NativeEnum.resolve(enumClass, getInt(0));
    }

    public int flags() {
        return getInt(0);
    }

}
