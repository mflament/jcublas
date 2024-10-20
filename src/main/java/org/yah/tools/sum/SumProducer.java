package org.yah.tools.sum;

/**
 * Sum float/double bits
 */
public interface SumProducer {

    int sum(float[] data, int offset, int length);

    default int sum(float[] data) {
        return sum(data, 0, data.length);
    }

    long sum(double[] data, int offset, int length);

    default long sum(double[] data) {
        return sum(data, 0, data.length);
    }

}
