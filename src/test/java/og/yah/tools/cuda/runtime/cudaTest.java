package og.yah.tools.cuda.runtime;

import com.sun.jna.Native;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;
import og.yah.tools.cuda.jna.size_t;
import org.junit.jupiter.api.Test;

class cudaTest {

    @Test
    void cudaRuntimeGetVersion() {
        RuntimeAPI cuda = RuntimeAPI.load();
        cuda.cudaSetDevice(0).check();
        IntByReference versionRef = new IntByReference();
        cuda.cudaRuntimeGetVersion(versionRef).check();
        int version = versionRef.getValue();
        System.out.printf("cuda version %d.%d%n", (version / 1000), (version % 1000));

        cuda.cudaDriverGetVersion(versionRef).check();
        version = versionRef.getValue();
        System.out.printf("driver version %d.%d%n", (version / 1000), (version % 1000));

        cudaDeviceProp.ByReference devProps = new cudaDeviceProp.ByReference();
        cuda.cudaGetDeviceProperties(devProps, 0).check();
        System.out.printf("%s (shader version %d.%d)%n", Native.toString(devProps.name), devProps.major, devProps.minor);

        size_t.ByReference free = new size_t.ByReference();
        size_t.ByReference total = new size_t.ByReference();
        cuda.cudaMemGetInfo(free, total).check();
        System.out.printf("memory %s/%s%n", free.getValue(), total.getValue());

        PointerByReference ptrRef = new PointerByReference();
        int size = 100 * 1024 * 1024;
        cuda.cudaMalloc(ptrRef, new size_t(size)).check();

        cuda.cudaMemGetInfo(free, total).check();
        System.out.printf("memory %s/%s%n", free.getValue(), total.getValue());

        cuda.cudaFree(ptrRef.getValue()).check();

        cuda.cudaMemGetInfo(free, total).check();
        System.out.printf("memory %s/%s%n", free.getValue(), total.getValue());
    }

}