package og.yah.tools.cuda.runtime;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;

@SuppressWarnings("unused")
@FunctionalInterface
public interface cudaStreamCallback_t extends Callback {
    void execute(cudaStream_t stream,  cudaError_t status, Pointer userData);
}
