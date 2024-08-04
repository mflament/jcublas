package og.yah.tools.cuda.runtime;

import com.sun.jna.Callback;
import com.sun.jna.Pointer;

@SuppressWarnings("unused")
@FunctionalInterface
public interface cudaHostFn_t extends Callback {
    void execute(Pointer useData);
}
