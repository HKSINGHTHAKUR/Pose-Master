package com.posemaster.app;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Locates the game's data files.
 *
 * <p>Resources are looked up on the classpath first, so the packaged jar works from any working
 * directory.  When running from a source checkout the sibling project directory is used, which
 * keeps the original repository layout working without duplicating the assets.
 */
public final class ResourceLocator {

    private final Path root;

    public ResourceLocator() {
        this(defaultRoot());
    }

    public ResourceLocator(Path root) {
        this.root = root;
    }

    private static Path defaultRoot() {
        String override = System.getProperty("posemaster.resources");
        if (override != null && !override.isBlank()) {
            return Paths.get(override);
        }
        // A packaged jar: resources live inside the jar.
        Path working = Paths.get("").toAbsolutePath();
        // A source checkout: the assets sit next to the pom.
        Path candidate = working.resolve("src/main/resources");
        if (Files.isDirectory(candidate)) {
            return candidate;
        }
        if (Files.isDirectory(working.resolve("resources"))) {
            return working.resolve("resources");
        }
        if (Files.isDirectory(working.resolve("Pose-Perfect"))) {
            return working.resolve("Pose-Perfect");
        }
        return working;
    }

    /** Returns the directory the resources are read from. */
    public Path root() {
        return root;
    }

    /** Opens a resource stream, falling back to the classpath. */
    public InputStream open(String name) throws IOException {
        Path file = root.resolve(name);
        if (Files.isRegularFile(file)) {
            return Files.newInputStream(file);
        }
        InputStream fromClasspath = ResourceLocator.class.getClassLoader().getResourceAsStream(name);
        if (fromClasspath == null) {
            throw new IOException("Resource not found: " + name);
        }
        return fromClasspath;
    }

    /** Resolves a resource to a file path, copying classpath resources out when needed. */
    public Path resolve(String name) throws IOException {
        Path file = root.resolve(name);
        if (Files.isRegularFile(file)) {
            return file;
        }
        try (InputStream in = open(name)) {
            Path cache = Files.createTempFile("posemaster", "-" + name.replace('/', '-'));
            Files.copy(in, cache, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            cache.toFile().deleteOnExit();
            return cache;
        }
    }

    /** Returns whether a resource exists. */
    public boolean exists(String name) {
        if (Files.isRegularFile(root.resolve(name))) {
            return true;
        }
        return ResourceLocator.class.getClassLoader().getResource(name) != null;
    }
}
