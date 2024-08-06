package org.yah.tools.cuda.cublas;

public class CublasException extends RuntimeException {
    private final cublasStatus_t status;

    public CublasException(cublasStatus_t status) {
        super("Cublas error " + status.nativeValue() + " (" + status.name() + ")");
        this.status = status;
    }

    public cublasStatus_t status() {
        return status;
    }
}
