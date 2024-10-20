package org.yah.tools.cuda.driver;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;

public interface CUoccupancyB2DSize extends Callback {
    Pointer getSharedMemorySize(int blockSize);
}
