package org.yah.tools.memcpy;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.runtime.RuntimeAPI;
import org.yah.tools.cuda.runtime.cudaMemcpyKind;
import org.yah.tools.memcpy.MemcpyKernelsSupport.MemcpyFunctionId;

public class CudaMemcpyBenchmark {

    public static final long GB = 1024 * 1024 * 1024;

    private static final long SIZE = 2 * GB;
    private static final int LENGTH = (int) (SIZE / Long.BYTES);
    private static final int WARMUPS = 3;
    private static final int RUNS = 5;
    private static final boolean TEST = false;

    private static int deviceOrdinal = 0;
    private static int blockSize = 64;

    private static final RuntimeAPI cuda = RuntimeAPI.load();

    private static void run() {
        cuda.cudaSetDevice(deviceOrdinal);
        MemcpyKernelsSupport kernelsSupport = new MemcpyKernelsSupport(cuda, deviceOrdinal, blockSize);
        Pointer hostMemory = new Memory(SIZE);
        Pointer deviceMemory = allocateDevice(SIZE);
        test(kernelsSupport, hostMemory, deviceMemory);
        bench(kernelsSupport, hostMemory, deviceMemory);
        kernelsSupport.close();
    }

    private static void test(MemcpyKernelsSupport kernelsSupport, Pointer hostMemory, Pointer deviceMemory) {
        long[] javaMemory = new long[LENGTH];
        test(kernelsSupport, hostMemory, deviceMemory, javaMemory, false, null);
        test(kernelsSupport, hostMemory, deviceMemory, javaMemory, true, null);
        for (MemcpyFunctionId functionId : MemcpyFunctionId.values()) {
            test(kernelsSupport, hostMemory, deviceMemory, javaMemory, true, functionId);
        }
    }

    private static void test(MemcpyKernelsSupport kernelsSupport, Pointer hostMemory, Pointer deviceMemory, long[] javaMemory,
                             boolean pinHostMemory, MemcpyFunctionId functionId) {
        if (!TEST)
            return;
        System.out.println("Testing " + label(pinHostMemory, functionId));
        for (int i = 0; i < javaMemory.length; i++) javaMemory[i] = i;
        hostMemory.write(0, javaMemory, 0, LENGTH);

        Pointer hostPtr = pinHostMemory ? kernelsSupport.hostRegister(hostMemory, SIZE) : hostMemory;
        run(kernelsSupport, hostPtr, deviceMemory, functionId);
        if (pinHostMemory) kernelsSupport.hostUnregister(hostMemory);

        hostMemory.read(0, javaMemory, 0, LENGTH);
        int errors = 0;
        for (int i = 0; i < javaMemory.length; i++) {
            if (javaMemory[i] != i * 2L) errors++;
        }
        if (errors > 0)
            throw new IllegalStateException(label(pinHostMemory, functionId) + ": " + errors + " errors");
    }

    private static void bench(MemcpyKernelsSupport kernelsSupport, Pointer hostMemory, Pointer deviceMemory) {
        if (RUNS == 0)
            return;
        bench(kernelsSupport, hostMemory, deviceMemory, false, null);
        bench(kernelsSupport, hostMemory, deviceMemory, true, null);
        for (MemcpyFunctionId functionId : MemcpyFunctionId.values())
            bench(kernelsSupport, hostMemory, deviceMemory, true, functionId);
    }

    private static void bench(MemcpyKernelsSupport kernelsSupport, Pointer hostMemory, Pointer deviceMemory, boolean pinHostMemory, MemcpyFunctionId functionId) {
        Pointer hostPtr = pinHostMemory ? kernelsSupport.hostRegister(hostMemory, SIZE) : hostMemory;

        for (int i = 0; i < WARMUPS; i++) run(kernelsSupport, hostPtr, deviceMemory, functionId);

        MemoryTimes memoryTimes = new MemoryTimes();
        long[] javaMemory = new long[LENGTH];
        for (int i = 0; i < RUNS; i++) {
            for (int j = 0; j < javaMemory.length; j++) javaMemory[j] = j;
            hostMemory.write(0, javaMemory, 0, LENGTH);
            memoryTimes.add(run(kernelsSupport, hostPtr, deviceMemory, functionId));
        }

        long totalSize = SIZE / GB * RUNS;
        System.out.printf("%-25s host2device=%.3f GB/s device2host=%.3f GB/s%n", label(pinHostMemory, functionId),
                totalSize / memoryTimes.host2device, totalSize / memoryTimes.device2host);

        if (pinHostMemory) kernelsSupport.hostUnregister(hostMemory);
    }

    private static final class MemoryTimes {
        double host2device;
        double device2host;

        public void add(MemoryTimes times) {
            this.host2device += times.host2device;
            this.device2host += times.device2host;
        }
    }

    private static MemoryTimes run(MemcpyKernelsSupport kernelsSupport, Pointer hostPtr, Pointer deviceMemory, MemcpyFunctionId functionId) {
        MemoryTimes memoryTimes = new MemoryTimes();

        long start = System.nanoTime();
        if (functionId != null)
            kernelsSupport.memcpy(functionId, deviceMemory, hostPtr, LENGTH);
        else
            cuda.cudaMemcpy(deviceMemory, hostPtr, SIZE, cudaMemcpyKind.cudaMemcpyHostToDevice).check();
        memoryTimes.host2device += (System.nanoTime() - start) * 1E-9;

        kernelsSupport.scale(deviceMemory, LENGTH, 2);

        start = System.nanoTime();
        if (functionId != null)
            kernelsSupport.memcpy(functionId, hostPtr, deviceMemory, LENGTH);
        else
            cuda.cudaMemcpy(hostPtr, deviceMemory, SIZE, cudaMemcpyKind.cudaMemcpyDeviceToHost).check();
        memoryTimes.device2host += (System.nanoTime() - start) * 1E-9;

        return memoryTimes;
    }

    private static Pointer allocateDevice(long size) {
        PointerByReference ptrRef = new PointerByReference();
        cuda.cudaMalloc(ptrRef, size).check();
        return ptrRef.getValue();
    }

    private static String label(boolean pinHostMemory, MemcpyFunctionId functionId) {
        if (functionId != null)
            return "memcpy_" + functionId;
        String s = "cudaMemcpy";
        if (pinHostMemory) s += " (pin host)";
        return s;
    }

    public static void main(String[] args) {
        CudaMemcpyBenchmark.run();
    }
}
