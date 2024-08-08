package org.yah.tools.cuda.kernel;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.yah.tools.cuda.driver.CUresult;
import org.yah.tools.cuda.driver.DriverAPI;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.Objects;

public class KernelInvocationHandler {

    private final DriverAPI driverAPI;
    private final Pointer functionPointer;
    private final long[] parameterOffsets;
    private final ParameterWriter[] parameterWriters;
    private final Pointer[] parameterPointers;
    private final Memory parametersMemory;

    public KernelInvocationHandler(DriverAPI driverAPI, Pointer functionPointer, Method method) {
        this.driverAPI = Objects.requireNonNull(driverAPI, "driverAPI is null");
        this.functionPointer = Objects.requireNonNull(functionPointer, "functionPointer is null");
        Parameter[] parameters = method.getParameters();
        if (parameters.length == 0 || !parameters[0].getType().isAssignableFrom(ExecutionConfig.class))
            throw new IllegalArgumentException("First parameter of kernel method " + method + " must be an instance of " + ExecutionConfig.class.getName());
        if (!CUresult.class.isAssignableFrom(method.getReturnType()))
            throw new IllegalArgumentException("return type of kernel method " + method + " must be an instance of " + CUresult.class.getName());

        int kernelParameterCount = parameters.length - 1;
        parameterOffsets = new long[kernelParameterCount];
        parameterWriters = new ParameterWriter[kernelParameterCount];
        long size = 0;
        for (int i = 0; i < kernelParameterCount; i++) {
            Parameter parameter = parameters[i + 1];
            Class<?> type = parameter.getType();
            parameterOffsets[i] = size;
            if (type == byte.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setByte(offset, (byte) value);
                size += 1;
            } else if (type == short.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setShort(offset, (short) value);
                size += Short.BYTES;
            } else if (type == int.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setInt(offset, (int) value);
                size += Integer.BYTES;
            } else if (type == long.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setLong(offset, (long) value);
                size += Long.BYTES;
            } else if (type == float.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setFloat(offset, (float) value);
                size += Float.BYTES;
            } else if (type == double.class) {
                parameterWriters[i] = (dst, offset, value) -> dst.setDouble(offset, (double) value);
                size += Double.BYTES;
            } else {
                if (Pointer.class.isAssignableFrom(type)) {
                    parameterWriters[i] = (dst, offset, value) -> dst.setPointer(offset, (Pointer) value);
                    size += Native.POINTER_SIZE;
                } else
                    throw new IllegalArgumentException(String.format("Unsupported parameter %s type %s for %s", parameter.getName(), parameter.getType(), method.getName()));
            }
        }
        parameterPointers = new Pointer[kernelParameterCount];
        if (size > 0) {
            parametersMemory = new Memory(size);
            for (int i = 0; i < kernelParameterCount; i++) {
                parameterPointers[i] = parametersMemory.share(parameterOffsets[i]);
            }
        } else {
            parametersMemory = null;
        }
    }

    public CUresult invoke(Object[] args) {
        ExecutionConfig executionConfig = (ExecutionConfig) args[0];
        Objects.requireNonNull(executionConfig, "executionConfig is null");
        for (int i = 0; i < parameterWriters.length; i++) {
            parameterWriters[i].write(parametersMemory, parameterOffsets[i], args[i + 1]);
        }
        return driverAPI.cuLaunchKernel(functionPointer, executionConfig.gridDim().x, executionConfig.gridDim().y, executionConfig.gridDim().z,
                executionConfig.blockDim().x, executionConfig.blockDim().y, executionConfig.blockDim().z, (int) executionConfig.shared(),
                executionConfig.stream(), parameterPointers, null);
    }

    @FunctionalInterface
    private interface ParameterWriter {
        void write(Memory dst, long offset, Object value);
    }
}
