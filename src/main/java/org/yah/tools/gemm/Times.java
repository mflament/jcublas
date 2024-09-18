package org.yah.tools.gemm;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

@SuppressWarnings("unused")
public final class Times {

    public enum Operation {
        TRANSPOSE,
        WRITE,
        GEMM,
        READ;

        @Override
        public String toString() {
            return name().toLowerCase();
        }
    }

    private final String name;
    private final Map<Operation, Time> times = new EnumMap<>(Operation.class);

    public Times(String name) {
        this.name = name;
    }

    public Times(Times from) {
        this.name = from.name;
        from.times.forEach((name, time) -> this.times.put(name, new Time(time)));
    }

    public String name() {
        return name;
    }

    public double get(Operation operation) {
        Time time = times.get(operation);
        return time == null ? 0 : time.avg();
    }

    public double total() {
        return times.values().stream().mapToDouble(Time::avg).sum();
    }

    public void addMillis(Operation operation, double ms) {
        times.computeIfAbsent(operation, Time::new).addMillis(ms);
    }

    public void addNanos(Operation operation, double ns) {
        times.computeIfAbsent(operation, Time::new).addNanos(ns);
    }

    public void measure(Operation operation, Runnable task) {
        long start = System.nanoTime();
        task.run();
        addNanos(operation, System.nanoTime() - start);
    }

    public void reset() {
        times.values().forEach(Time::reset);
    }

    @Override
    public String toString() {
        if (times.size() == 1)
            return times.values().iterator().next().toString();

        double sum = times.values().stream().mapToDouble(Time::avg).sum();
        StringBuilder sb = new StringBuilder();
        for (Time time : times.values()) {
            sb.append(String.format("%s=%.0fms (%.0f%%)", time.name(), time.avg(), time.avg() / sum * 100)).append(" ");
        }
        sb.append(String.format("total=%.0fms", sum));
        return sb.toString();
    }

    public static final class Time {
        private final Operation operation;
        private double totalMs;
        private int runs;

        public Time(Operation operation) {
            this.operation = Objects.requireNonNull(operation, "operation is null");
        }

        public Time(Time from) {
            this.operation = from.operation;
            this.totalMs = from.totalMs;
            this.runs = from.runs;
        }

        public Operation name() {
            return operation;
        }

        public double totalMs() {
            return totalMs;
        }

        public int runs() {
            return runs;
        }

        private void addMillis(double ms) {
            totalMs += ms;
            runs++;
        }

        private void addNanos(double ns) {
            totalMs += ns * 1E-6;
            runs++;
        }

        public boolean hasRun() {
            return runs > 0;
        }

        public double avg() {
            return totalMs / runs;
        }

        public void reset() {
            runs = 0;
            totalMs = 0;
        }

        @Override
        public String toString() {
            return String.format("%s=%.1fms", operation, totalMs / runs);
        }
    }

}
