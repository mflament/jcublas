package org.yah.tools.cuda.runtime;

@SuppressWarnings("unused")
public class CudaException extends RuntimeException {
    private final cudaError_t error;

    public CudaException(cudaError_t error) {
        super("cuda error " + error.nativeValue() + " (" + error.name() + ")");
        this.error = error;
    }

    public cudaError_t error() {
        return error;
    }
}
