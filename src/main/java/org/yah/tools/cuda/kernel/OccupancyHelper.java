package org.yah.tools.cuda.kernel;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.driver.DriverAPI;

public final class OccupancyHelper {
    private OccupancyHelper() {
    }

    public static int getMaxActiveBlocksPerMultiprocessor(DriverAPI cu, Pointer module, String functionName, int blockSize, long dynamicSMemSize) {
        PointerByReference ptrRef = new PointerByReference();
        cu.cuModuleGetFunction(ptrRef, module, functionName).check();
        Pointer func = ptrRef.getValue();
        IntByReference numBlocksPtr = new IntByReference();
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocksPtr, func, blockSize, Pointer.createConstant(dynamicSMemSize)).check();
        return numBlocksPtr.getValue();
    }

}
