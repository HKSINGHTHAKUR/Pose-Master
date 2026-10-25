package com.posemaster.app;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HighScoreStoreTest {

    @Test
    void savesAndLoadsTheHighScore(@TempDir Path directory) throws IOException {
        Path file = directory.resolve("high_score.txt");
        HighScoreStore store = new HighScoreStore(file);

        store.save(42);
        assertEquals(42, store.load(0));
    }

    @Test
    void fallsBackToTheSeedWhenNothingIsStored(@TempDir Path directory) {
        HighScoreStore store = new HighScoreStore(directory.resolve("high_score.txt"));
        assertEquals(7, store.load(7));
    }

    @Test
    void ignoresACorruptScoreFile(@TempDir Path directory) throws IOException {
        Path file = directory.resolve("high_score.txt");
        Files.writeString(file, "not a number");

        HighScoreStore store = new HighScoreStore(file);
        assertEquals(3, store.load(3));
    }

    @Test
    void createsTheParentDirectory(@TempDir Path directory) throws IOException {
        Path file = directory.resolve("nested").resolve("deeper").resolve("high_score.txt");
        HighScoreStore store = new HighScoreStore(file);

        store.save(11);

        assertEquals(11, store.load(0));
        assertEquals("11", Files.readString(file));
    }
}
