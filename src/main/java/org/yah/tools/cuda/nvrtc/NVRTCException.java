package org.yah.tools.cuda.nvrtc;

@SuppressWarnings("unused")
public class NVRTCException extends RuntimeException {
    private final nvrtcResult result;

    public NVRTCException(nvrtcResult result) {
        super("NVRTC error " + result.nativeValue() + " (" + result.name() + ")");
        this.result = result;
    }

    public nvrtcResult result() {
        return result;
    }
}
