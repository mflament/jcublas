package org.yah.tools.memcpy;

import com.sun.jna.Memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@SuppressWarnings({"PointlessArithmeticExpression", "ConstantValue"})
public class HostMemcpyBenchmark {

    public static final long GB = 1024 * 1024 * 1024;

    // must be <= 2 GB to test ByteBuffer
    private static final long SIZE = 1 * GB;
    private static final int LENGTH = (int) (SIZE / Long.BYTES);
    private static final int WARMUPS = 3;
    private static final int RUNS = 5;
    private static final boolean TEST = false;
    private static final int MAX_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int[] THREAD_COUNTS = {0, 2, 4, 8};

    private static final ExecutorService executorService = Executors.newFixedThreadPool(MAX_THREADS);

    public static void bench() {
        long[] javaMemory = new long[LENGTH];
        try {
            MemoryBenchmark benchmark = new JNABenchmark(SIZE);
            run(javaMemory, benchmark);
            benchmark.close();

            benchmark = new ForeignBenchmark(SIZE);
            run(javaMemory, benchmark);
            benchmark.close();

            if (SIZE <= Integer.MAX_VALUE) {
                benchmark = new ByteBufferBenchmark(SIZE);
                run(javaMemory, benchmark);
                benchmark.close();
            }
        } finally {
            executorService.shutdownNow();
        }
    }

    private static void run(long[] javaMemory, MemoryBenchmark benchmark) {
        System.out.println(benchmark);
        warmup(javaMemory, benchmark);

        for (int threadCount : THREAD_COUNTS) {
            test(javaMemory, benchmark, threadCount);
            bench(javaMemory, benchmark, threadCount);
        }
    }

    private static void test(long[] javaMemory, MemoryBenchmark benchmark, int threadCount) {
        if (!TEST)
            return;
        for (int i = 0; i < javaMemory.length; i++) javaMemory[i] = i;
        benchmark.write(javaMemory, threadCount);

        long[] copiedMemory = new long[javaMemory.length];
        benchmark.read(copiedMemory, threadCount);

        int errors = 0;
        for (int i = 0; i < javaMemory.length; i++) {
            if (javaMemory[i] != copiedMemory[i])
                errors++;
        }
        if (errors > 0)
            throw new IllegalStateException(errors + " errors");
    }

    private static void warmup(long[] javaMemory, MemoryBenchmark benchmark) {
        if (WARMUPS == 0)
            return;
        for (int i = 0; i < WARMUPS; i++) {
            benchmark.write(javaMemory, 0);
            benchmark.read(javaMemory, 0);
            benchmark.write(javaMemory, 2);
            benchmark.read(javaMemory, 2);
        }
    }

    private static void bench(long[] javaMemory, MemoryBenchmark benchmark, int threadCount) {
        if (RUNS == 0)
            return;
        double totalWriteSecs = 0, totalReadSecs = 0;
        for (int i = 0; i < RUNS; i++) {
            long start = System.nanoTime();
            benchmark.write(javaMemory, threadCount);
            totalWriteSecs += (System.nanoTime() - start) * 1E-9;

            start = System.nanoTime();
            benchmark.read(javaMemory, threadCount);
            totalReadSecs += (System.nanoTime() - start) * 1E-9;
        }
        long totalSize = SIZE / GB * RUNS;
        System.out.printf(" %d threads : java2host=%.3f GB/s host2java=%.3f GB/s%n", threadCount, totalSize / totalWriteSecs, totalSize / totalReadSecs);
    }

    private interface MemoryBenchmark extends AutoCloseable {
        void write(long[] javaMemory, int threadCount);

        void read(long[] javaMemory, int threadCount);

        @Override
        void close();
    }

    private interface ChunkTask {
        void run(int index, int length);
    }

    private static final class JNABenchmark implements MemoryBenchmark {
        private final Memory hostMemory;

        public JNABenchmark(long size) {
            hostMemory = new Memory(size);
        }

        @Override
        public void write(long[] javaMemory, int threadCount) {
            run(javaMemory, threadCount, (index, length) -> hostMemory.write(index * (long) Long.BYTES, javaMemory, index, length));
        }

        @Override
        public void read(long[] javaMemory, int threadCount) {
            run(javaMemory, threadCount, (index, length) -> hostMemory.read(index * (long) Long.BYTES, javaMemory, index, length));
        }

        @Override
        public void close() {
            hostMemory.close();
        }

        @Override
        public String toString() {
            return "JNA";
        }
    }

    private static final class ForeignBenchmark implements MemoryBenchmark {
        private final Arena arena;
        private final MemorySegment hostMemory;

        public ForeignBenchmark(long size) {
            arena = Arena.ofShared();
            hostMemory = arena.allocate(size);
        }

        @Override
        public void write(long[] javaMemory, int threadCount) {
            MemorySegment src = MemorySegment.ofArray(javaMemory);
            run(javaMemory, threadCount, (index, length) -> memcpy(hostMemory, src, index, length));
        }

        @Override
        public void read(long[] javaMemory, int threadCount) {
            MemorySegment dst = MemorySegment.ofArray(javaMemory);
            run(javaMemory, threadCount, (index, length) -> memcpy(dst, hostMemory, index, length));
        }

        private static void memcpy(MemorySegment dst, MemorySegment src, int index, int length) {
            long offset = index * (long) Long.BYTES;
            long size = length * (long) Long.BYTES;
            dst.asSlice(offset, size).copyFrom(src.asSlice(offset, size));
        }

        @Override
        public void close() {
            arena.close();
        }

        @Override
        public String toString() {
            return "Foreign";
        }
    }

    private static final class ByteBufferBenchmark implements MemoryBenchmark {
        private ByteBuffer byteBuffer;
        private LongBuffer longBuffer;

        public ByteBufferBenchmark(long size) {
            if (size > Integer.MAX_VALUE)
                throw new IllegalArgumentException(size + " overflow direct byte buffer max size " + Integer.MAX_VALUE);
            byteBuffer = ByteBuffer.allocateDirect((int) size).order(ByteOrder.LITTLE_ENDIAN);
            longBuffer = byteBuffer.asLongBuffer();
        }

        @Override
        public void write(long[] javaMemory, int threadCount) {
            run(javaMemory, threadCount, (index, length) -> write(javaMemory, index, length));
        }

        @Override
        public void read(long[] javaMemory, int threadCount) {
            run(javaMemory, threadCount, (index, length) -> read(javaMemory, index, length));
        }

        private void write(long[] src, int index, int length) {
            LongBuffer srcBuffer = LongBuffer.wrap(src, index, length);
            longBuffer.slice(index, length).put(srcBuffer);
        }

        private void read(long[] dst, int index, int length) {
            LongBuffer dstBuffer = LongBuffer.wrap(dst, index, length);
            dstBuffer.put(longBuffer.slice(index, length));
        }

        @Override
        public void close() {
            longBuffer = null;
            byteBuffer = null;
        }

        @Override
        public String toString() {
            return "ByteBuffer";
        }
    }

    private static void run(long[] javaMemory, int threadCount, ChunkTask callback) {
        if (threadCount == 0) {
            callback.run(0, javaMemory.length);
        } else {
            int chunkSize = (int) Math.ceil(javaMemory.length / (double) threadCount);
            CountDownLatch latch = new CountDownLatch(threadCount);
            for (int i = 0; i < threadCount; i++) {
                int index = i * chunkSize;
                int chunkLength = Math.min(chunkSize, javaMemory.length - index);
                executorService.submit(() -> {
                    callback.run(index, chunkLength);
                    latch.countDown();
                });
            }
            try {
                latch.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static void main(String[] args) {
        HostMemcpyBenchmark.bench();
    }
}
