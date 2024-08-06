package org.yah.tools.gemm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

@SuppressWarnings("unused")
public final class Times {

    public static final class Time {
        private final String name;
        private double totalMs;
        private int runs;

        public Time(String name) {
            this.name = Objects.requireNonNull(name, "name is null");
        }

        public String name() {
            return name;
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

        @Override
        public String toString() {
            return String.format("%s=%.1fms", name, totalMs / runs);
        }
    }

    private final Map<String, Time> times = new LinkedHashMap<>();

    public void addMillis(String name, double ms) {
        times.computeIfAbsent(name, Time::new).addMillis(ms);
    }

    public void addNanos(String name, double ns) {
        times.computeIfAbsent(name, Time::new).addNanos(ns);
    }

    public void measure(String name, Runnable task) {
        long start = System.nanoTime();
        task.run();
        addNanos(name, System.nanoTime() - start);
    }

    @Override
    public String toString() {
        if (times.size() == 1)
            return times.values().iterator().next().toString();

        double sum = times.values().stream().mapToDouble(Time::avg).sum();
        StringBuilder sb = new StringBuilder();
        for (Time time : times.values()) {
            sb.append(String.format("%s=%.1fms (%.0f%%)", time.name(), time.avg(), time.avg() / sum * 100)).append(" ");
        }
        sb.append(String.format("total=%.1f", sum));
        return sb.toString();
    }
}
