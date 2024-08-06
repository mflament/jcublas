package org.yah.tools.cuda.jna;

import com.sun.jna.FromNativeContext;
import com.sun.jna.NativeMapped;

public interface NativeEnum extends NativeMapped {

    int ordinal();

    @SuppressWarnings("unused")
    String name();

    default int nativeValue() {
        return ordinal();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    @Override
    default Object fromNative(Object nativeValue, FromNativeContext context) {
        return resolve((Class) context.getTargetType(), (int) nativeValue);
    }

    @Override
    default Object toNative() {
        return nativeValue();
    }

    @Override
    default Class<?> nativeType() {
        return int.class;
    }

    static <E extends Enum<E> & NativeEnum> E resolve(Class<E> enumClass, int value) {
        E[] enumConstants = enumClass.getEnumConstants();
        for (E enumConstant : enumConstants) {
            if (enumConstant.nativeValue() == value)
                return enumConstant;
        }
        throw new IllegalArgumentException("unresolved enum '" + enumClass.getName() + "' constant for native value " + value);
    }

    @SuppressWarnings("unchecked")
    static <E extends NativeEnum> int flags(E... enums) {
        int flags = 0;
        for (E e : enums)
            flags |= e.nativeValue();
        return flags;
    }

}
