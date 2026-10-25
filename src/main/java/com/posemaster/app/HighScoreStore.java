package com.posemaster.app;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Stores the player's high score.
 *
 * <p>The original Python game kept the score in {@code ~/.pose_master/high_score.txt}, seeding
 * it from a bundled copy on first run.  That behaviour is preserved here.
 */
public final class HighScoreStore {

    /** Directory the original game used for per-user data. */
    public static final String USER_DIRECTORY = ".pose_master";
    /** File name of the high score file. */
    public static final String FILE_NAME = "high_score.txt";

    private final Path file;

    public HighScoreStore() {
        this(defaultPath());
    }

    public HighScoreStore(Path file) {
        this.file = file;
    }

    private static Path defaultPath() {
        return Paths.get(System.getProperty("user.home"), USER_DIRECTORY, FILE_NAME);
    }

    /** Returns the file the score is written to. */
    public Path file() {
        return file;
    }

    /**
     * Reads the stored high score.
     *
     * @param seedValue the value to use when no score has been stored yet
     */
    public int load(int seedValue) {
        try {
            if (Files.isRegularFile(file)) {
                String text = Files.readString(file, StandardCharsets.UTF_8).trim();
                if (!text.isEmpty()) {
                    return Integer.parseInt(text);
                }
            }
        } catch (IOException | NumberFormatException e) {
            // A corrupt or unreadable score file should not stop the game.
        }
        return seedValue;
    }

    /** Writes the high score, creating the directory if needed. */
    public void save(int score) throws IOException {
        Path parent = file.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.writeString(file, Integer.toString(score), StandardCharsets.UTF_8);
    }
}
