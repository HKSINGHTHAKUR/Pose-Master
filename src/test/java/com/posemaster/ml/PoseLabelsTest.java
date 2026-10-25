package com.posemaster.ml;

import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoseLabelsTest {

    private static Path labelsFile() {
        Path direct = Paths.get("src/main/resources/labels.npy");
        if (Files.isRegularFile(direct)) {
            return direct;
        }
        return Paths.get("labels.npy");
    }

    @Test
    void readsTheSixteenOriginalLabels() throws Exception {
        try (InputStream in = Files.newInputStream(labelsFile())) {
            List<String> labels = PoseLabels.load(in);
            assertEquals(16, labels.size());
            assertEquals("Crossed arms", labels.get(0));
            assertEquals("Tpose", labels.get(9));
            assertEquals("warriorpose", labels.get(15));
            for (String label : labels) {
                assertFalse(label.isBlank(), "labels must not contain blanks");
            }
        }
    }

    @Test
    void everyLabelHasAMatchingReferenceImage() throws Exception {
        try (InputStream in = Files.newInputStream(labelsFile())) {
            List<String> labels = PoseLabels.load(in);
            for (String label : labels) {
                Path image = Paths.get("src", "main", "resources", "poseImages", label + ".png");
                assertTrue(Files.isRegularFile(image), "missing reference image for " + label);
            }
        }
    }
}
