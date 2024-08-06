package org.yah.tools.gemm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntFunction;

public final class MatrixExecutor implements AutoCloseable {
    private final int concurrency;
    private final ExecutorService executor;

    public MatrixExecutor() {
        this(Runtime.getRuntime().availableProcessors());
    }

    public MatrixExecutor(int concurrency) {
        this.concurrency = concurrency;
        this.executor = Executors.newFixedThreadPool(concurrency);
    }

    @Override
    public void close() {
        executor.shutdown();
    }

    public void parallelize(int rows, int cols, Handler handler)  {
        parallelize(rows, cols, ignored -> handler);
    }

    public void parallelize(int rows, int cols, IntFunction<Handler> handlerFactory)  {
        int chunkSize = ceil_div(cols, concurrency);
        List<Future<?>> futures = new ArrayList<>(concurrency);
        // TOSEE : replace with StructuredConcurrency https://openjdk.org/jeps/462 when released ... or not
        for (int i = 0; i < concurrency; i++) {
            int from = i * chunkSize;
            int to = Math.min(cols, from + chunkSize);
            futures.add(executor.submit(new ColMajorTask(handlerFactory, i, from, to, rows)));
        }
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public int threads() {
        return concurrency;
    }

    private record ColMajorTask(IntFunction<Handler> handlerFactory, int threadIndex,
                                int fromCol, int toCol, int rows) implements Runnable {
        public void run() {
            Handler handler = handlerFactory.apply(threadIndex);
            for (int col = fromCol; col < toCol; ++col) {
                for (int row = 0; row < rows; ++row) {
                    handler.handle(row, col);
                }
            }
        }
    }

    @FunctionalInterface
    public interface Handler {
        void handle(int row, int col);
    }

    public static int ceil_div(int a, int b) {
        return (a + (b - 1)) / b;
    }
}
