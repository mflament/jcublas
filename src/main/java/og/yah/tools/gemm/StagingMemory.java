package og.yah.tools.gemm;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

public class StagingMemory {

    private Memory memory;

    public Pointer get(long size) {
        if (memory == null || memory.size() < size) {
            if (memory != null)
                memory.close();
            memory = new Memory(size);
        }
        return memory;
    }

    public void free() {
        if (memory != null) {
            memory.close();
            memory = null;
        }
    }

}
