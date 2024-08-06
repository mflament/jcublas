package org.yah.tools.cuda.jna;

import com.sun.jna.Platform;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public final class CudaLibrarySupport {
    private CudaLibrarySupport() {
    }

    public static String lookupLibrary(String name) {
        if (Platform.isWindows()) {
            Path cudaPath = Paths.get(System.getenv("CUDA_PATH")).resolve("bin");
            if (!Files.exists(cudaPath))
                throw new IllegalStateException(cudaPath + " not found");
            try (Stream<Path> files = Files.list(cudaPath)) {
                Path libraryPath = files.filter(f -> Files.isRegularFile(f) && f.getFileName().toString().startsWith(name)).findFirst()
                        .orElseThrow(() -> new IllegalStateException("No library " + name + " found in " + cudaPath));
                return libraryPath.toAbsolutePath().toString();
            } catch (IOException e) {
                throw new IllegalStateException("Error listing files in " + cudaPath, e);
            }
        } else {
            return name;
        }
    }
}
