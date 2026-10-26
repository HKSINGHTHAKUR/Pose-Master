package com.posemaster.ml;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxPoseClassifierTest {

    private static Path resource(String name) {
        Path direct = Paths.get("src/main/resources", name);
        if (Files.isRegularFile(direct)) {
            return direct;
        }
        Path packaged = Paths.get("target/classes", name);
        if (Files.isRegularFile(packaged)) {
            return packaged;
        }
        throw new IllegalStateException("Resource not found: " + name);
    }

    @Test
    void loadsTheModelAndItsLabels() throws Exception {
        try (OnnxPoseClassifier classifier = OnnxPoseClassifier.create(
                resource("models/pose_classifier.onnx"), resource("labels.txt"))) {

            List<String> labels = classifier.labels();
            assertEquals(16, labels.size());
            assertEquals("Crossed arms", labels.get(0));
            assertEquals("warriorpose", labels.get(15));
        }
    }

    @Test
    void predictsAValidLabelForAnyInput() throws Exception {
        try (OnnxPoseClassifier classifier = OnnxPoseClassifier.create(
                resource("models/pose_classifier.onnx"), resource("labels.txt"))) {

            float[] features = new float[OnnxPoseClassifier.FEATURE_COUNT];
            String label = classifier.classify(features);

            assertNotNull(label);
            assertTrue(classifier.labels().contains(label));
        }
    }

    @Test
    void probabilitiesFormAValidDistribution() throws Exception {
        try (OnnxPoseClassifier classifier = OnnxPoseClassifier.create(
                resource("models/pose_classifier.onnx"), resource("labels.txt"))) {

            float[] probabilities = classifier.predict(new float[OnnxPoseClassifier.FEATURE_COUNT]);

            assertEquals(16, probabilities.length);
            float sum = 0;
            for (float p : probabilities) {
                assertTrue(p >= 0f && p <= 1f, "probability out of range: " + p);
                sum += p;
            }
            assertEquals(1f, sum, 1e-4f);
        }
    }

    @Test
    void theArgmaxLabelMatchesTheClassification() throws Exception {
        try (OnnxPoseClassifier classifier = OnnxPoseClassifier.create(
                resource("models/pose_classifier.onnx"), resource("labels.txt"))) {

            float[] features = new float[OnnxPoseClassifier.FEATURE_COUNT];
            for (int i = 0; i < features.length; i++) {
                features[i] = (float) (Math.sin(i * 0.37) * 0.4);
            }
            float[] probabilities = classifier.predict(features);

            int best = 0;
            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > probabilities[best]) {
                    best = i;
                }
            }
            assertEquals(classifier.labels().get(best), classifier.classify(features));
        }
    }
}
