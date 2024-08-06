package org.yah.tools.cuda.driver;

@SuppressWarnings("unused")
public class DriverException extends RuntimeException {
    private final CUresult result;

    public DriverException(CUresult result) {
        super("DriverAPI error " + result.nativeValue() + " (" + result.name() + ")");
        this.result = result;
    }

    public CUresult result() {
        return result;
    }
}
