package org.yah.tools.cuda.runtime;

import org.yah.tools.cuda.jna.NativeEnum;

@SuppressWarnings("unused")
public enum cudaMemcpyKind implements NativeEnum {
    /**
     * < Host   -> Host
     */
    cudaMemcpyHostToHost(0),
    /**
     * < Host   -> Device
     */
    cudaMemcpyHostToDevice(1),
    /**
     * < Device -> Host
     */
    cudaMemcpyDeviceToHost(2),
    /**
     * < Device -> Device
     */
    cudaMemcpyDeviceToDevice(3),
    /**
     * < Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing
     */
    cudaMemcpyDefaultDevice(4);

    private final int nativeValue;

    cudaMemcpyKind(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
