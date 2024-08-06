package org.yah.tools.cuda.kernel;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.driver.CUdevice_attribute;
import org.yah.tools.cuda.driver.CUresult;
import org.yah.tools.cuda.driver.DriverAPI;
import org.yah.tools.cuda.nvrtc.NVRTC;
import org.yah.tools.cuda.nvrtc.nvrtcResult;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KernelSupport {

    private static final Logger LOGGER = LoggerFactory.getLogger(KernelSupport.class);

    private static final Collection<String> DEFAULT_OPTIONS = List.of("-rdc=true");

    private final RuntimeAPI runtimeAPI;
    private final DriverAPI driverAPI;
    private final NVRTC nvrtc;

    private final Pointer device;
    private final String computeCapability;

    // temporary variables
    private final PointerByReference ptrRef = new PointerByReference();

    public KernelSupport(RuntimeAPI runtimeAPI, int deviceOrdinal) {
        this(runtimeAPI, deviceOrdinal, DriverAPI.load(), NVRTC.load());
    }

    public KernelSupport(RuntimeAPI runtimeAPI, int deviceOrdinal, DriverAPI driverAPI, NVRTC nvrtc) {
        this.runtimeAPI = runtimeAPI;
        this.driverAPI = driverAPI;
        this.nvrtc = nvrtc;

        driverAPI.cuDeviceGet(ptrRef, deviceOrdinal).check();
        device = ptrRef.getValue();

        IntByReference attrRef = new IntByReference();
        driverAPI.cuDeviceGetAttribute(attrRef, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).check();
        int ccMajor = attrRef.getValue();
        driverAPI.cuDeviceGetAttribute(attrRef, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).check();
        int ccMinor = attrRef.getValue();
        computeCapability = String.format("%d%d", ccMajor, ccMinor);
        if (LOGGER.isDebugEnabled()) {
            int maxNameSize = 256;
            String name;
            try (Memory namePtr = new Memory(maxNameSize)) {
                driverAPI.cuDeviceGetName(namePtr, maxNameSize, device).check();
                name = namePtr.getString(0, "US_ASCII");
            }
            LOGGER.debug("Created KernelSupport for device {} (name={} computeCapability={})", deviceOrdinal, name, computeCapability);
        }
    }

    public Memory getPTX(Pointer program) {
        LongByReference sizeRef = new LongByReference();
        nvrtc.nvrtcGetPTXSize(program, sizeRef).check();
        long size = sizeRef.getValue();
        Memory ptx = new Memory(size);
        nvrtc.nvrtcGetPTX(program, ptx).check();
        return ptx;
    }

    public Memory getCUBIN(Pointer program) {
        LongByReference sizeRef = new LongByReference();
        nvrtc.nvrtcGetCUBINSize(program, sizeRef).check();
        long size = sizeRef.getValue();
        Memory cubin = new Memory(size);
        nvrtc.nvrtcGetCUBIN(program, cubin).check();
        return cubin;
    }

    public void destroyProgram(Pointer program) {
        ptrRef.setValue(program);
        nvrtc.nvrtcDestroyProgram(ptrRef).check();
    }

    /**
     * @param source kernel source
     * @return Pointer to NVRTCProgram
     */
    public Pointer compile(String source, Map<String, String> defines, String... options) {
        nvrtc.nvrtcCreateProgram(ptrRef, source, null, 0, null, null).check();
        Pointer program = ptrRef.getValue();
        List<String> optionsList = new ArrayList<>(Arrays.asList(options));
        if (optionsList.isEmpty())
            optionsList.addAll(DEFAULT_OPTIONS);
        optionsList.add("-arch=sm_" + computeCapability);
        defines.forEach((name, value) -> optionsList.add(formatDefine(name, value)));

        nvrtcResult compileResult = nvrtc.nvrtcCompileProgram(program, optionsList.size(), optionsList.toArray(String[]::new));

        LongByReference sizeRef = new LongByReference();
        nvrtc.nvrtcGetProgramLogSize(program, sizeRef).check();
        String programLogs = null;
        long size = sizeRef.getValue();
        if (size > 1) {
            try (Memory logsPtr = new Memory(size)) {
                nvrtc.nvrtcGetProgramLog(program, logsPtr).check();
                programLogs = logsPtr.getString(0, "US_ASCII");
            }
        }
        if (programLogs != null)
            LOGGER.info("NVRTC program compile logs:\n{}", programLogs);
        compileResult.check();
        return program;
    }

    public Pointer getPrimaryContext() {
        driverAPI.cuDevicePrimaryCtxRetain(ptrRef, device).check();
        return ptrRef.getValue();
    }

    public void releasePrimaryContext() {
        driverAPI.cuDevicePrimaryCtxRelease(device).check();
    }

    public Pointer loadModule(Pointer context, Memory image) {
        driverAPI.cuModuleLoadData(ptrRef, image).check();
        return ptrRef.getValue();
    }

    public void unloadModule(Pointer module) {
        driverAPI.cuModuleUnload(module).check();
    }

    public record CUFunction(String name, Pointer pointer, List<CUFunctionParameter> parameters) {
    }

    public record CUFunctionParameter(long offset, long size) {
    }

    public List<CUFunction> enumerateFunctions(Pointer module) {
        IntByReference intRef = new IntByReference();
        driverAPI.cuModuleGetFunctionCount(intRef, module).check();
        int functionCount = intRef.getValue();
        List<CUFunction> functions = new ArrayList<>(functionCount);
        try (Memory functionPointers = new Memory(functionCount * (long) Native.POINTER_SIZE)) {
            driverAPI.cuModuleEnumerateFunctions(functionPointers, functionCount, module).check();
            for (int i = 0; i < functionCount; i++) {
                Pointer functionPointer = functionPointers.getPointer(i * (long) Native.POINTER_SIZE);
                functions.add(createFunction(functionPointer, null));
            }
        }
        return functions;
    }

    public CUFunction getKernelFunction(Pointer module, String name) {
        driverAPI.cuModuleGetFunction(ptrRef, module, name).check();
        return createFunction(ptrRef.getValue(), name);
    }

    public <T> T createProxy(Pointer module, Class<T> moduleInterface) {
        if (!moduleInterface.isInterface())
            throw new IllegalArgumentException(moduleInterface.getName() + " is not an interface");
        Map<String, KernelInvocationHandler> invocationHandlers = new HashMap<>();
        Method[] methods = moduleInterface.getMethods();
        for (Method method : methods) {
            String name = method.getName();
            if (invocationHandlers.containsKey(name))
                throw new IllegalArgumentException("Duplicate kernel method " + name + " in " + moduleInterface.getName());
            CUFunction kernelFunction = getKernelFunction(module, name);
            KernelInvocationHandler handler = new KernelInvocationHandler(driverAPI, kernelFunction.pointer(), method);

            invocationHandlers.put(name, handler);
        }
        //noinspection unchecked
        return (T) Proxy.newProxyInstance(ClassLoader.getSystemClassLoader(), new Class[]{moduleInterface}, new ModuleInvocationHandler(invocationHandlers));
    }

    private CUFunction createFunction(Pointer functionPointer, @Nullable String name) {
        if (name == null) {
            driverAPI.cuFuncGetName(ptrRef, functionPointer).check();
            name = ptrRef.getValue().getString(0, "US-ASCII");
        }
        int paramIndex = 0;
        LongByReference offset = new LongByReference();
        LongByReference size = new LongByReference();
        List<CUFunctionParameter> parameters = new ArrayList<>();
        while (paramIndex < 1000) {
            CUresult result = driverAPI.cuFuncGetParamInfo(functionPointer, paramIndex, offset, size);
            if (result == CUresult.CUDA_ERROR_INVALID_VALUE)
                break;
            result.check();
            parameters.add(new CUFunctionParameter(offset.getValue(), size.getValue()));
            paramIndex++;
        }
        return new CUFunction(name, functionPointer, parameters);
    }

    private static String formatDefine(String name, String value) {
        if (StringUtils.isBlank(value))
            return "-D" + name;
        return "-D" + name + "=" + value;
    }
}
