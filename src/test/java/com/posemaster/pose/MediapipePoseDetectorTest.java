package com.posemaster.pose;

import nu.pattern.OpenCV;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks the pose pipeline against known-good landmarks for a fixture image.
 *
 * <p>The expected values for {@code src/test/resources/pose.jpg} are stored in
 * {@code src/test/resources/expected_landmarks.json}.
 */
class MediapipePoseDetectorTest {

    private static final double TOLERANCE_PIXELS = 2.0;

    @BeforeAll
    static void loadOpenCv() {
        OpenCV.loadLocally();
    }

    private static Path resource(String name) {
        Path direct = Paths.get("src/test/resources", name);
        if (Files.isRegularFile(direct)) {
            return direct;
        }
        Path packaged = Paths.get("target/test-classes", name);
        if (Files.isRegularFile(packaged)) {
            return packaged;
        }
        throw new IllegalStateException("Test resource not found: " + name);
    }

    private static Path model(String name) {
        Path direct = Paths.get("src/main/resources/models", name);
        if (Files.isRegularFile(direct)) {
            return direct;
        }
        Path packaged = Paths.get("target/classes/models", name);
        if (Files.isRegularFile(packaged)) {
            return packaged;
        }
        throw new IllegalStateException("Model not found: " + name);
    }

    @Test
    void pipelineMatchesTheExpectedLandmarks() throws Exception {
        // Load the JPEG through OpenCV so the pixel data matches the one used for the expected landmarks.
        Mat frame = org.opencv.imgcodecs.Imgcodecs.imread(resource("pose.jpg").toString());
        assertNotNull(frame);
        assertTrue(frame.cols() > 0 && frame.rows() > 0);

        double[][] expected = readExpectedLandmarks();

        try (MediapipePoseDetector detector = MediapipePoseDetector.create(
                model("person_detection_mediapipe_2023mar.onnx"),
                model("pose_estimation_mediapipe_2023mar.onnx"))) {

            PoseLandmarks landmarks = detector.detect(frame);
            assertNotNull(landmarks, "the detector should find the person in the test photo");

            for (int i = 0; i < PoseLandmarks.LANDMARK_COUNT; i++) {
                PoseLandmark actual = landmarks.get(i);
                double expectedX = expected[i][0];
                double expectedY = expected[i][1];
                // The detector reports normalised coordinates; the fixture stores pixel ones.
                double actualX = actual.x() * frame.cols();
                double actualY = actual.y() * frame.rows();

                assertTrue(Math.abs(actualX - expectedX) < TOLERANCE_PIXELS,
                        "landmark " + i + " x: expected " + expectedX + " but was " + actualX);
                assertTrue(Math.abs(actualY - expectedY) < TOLERANCE_PIXELS,
                        "landmark " + i + " y: expected " + expectedY + " but was " + actualY);
            }
        }
    }

    @Test
    void theBodyIsReportedAsFullyVisibleInTheTestPhoto() throws Exception {
        Mat frame = readFrame();
        try (MediapipePoseDetector detector = MediapipePoseDetector.create(
                model("person_detection_mediapipe_2023mar.onnx"),
                model("pose_estimation_mediapipe_2023mar.onnx"))) {
            PoseLandmarks landmarks = detector.detect(frame);
            assertNotNull(landmarks);
            assertTrue(landmarks.isBodyFullyVisible(),
                    "the whole body is visible in the test photo");
        }
    }

    @Test
    void theFeatureVectorHasTheExpectedShape() throws Exception {
        Mat frame = readFrame();
        try (MediapipePoseDetector detector = MediapipePoseDetector.create(
                model("person_detection_mediapipe_2023mar.onnx"),
                model("pose_estimation_mediapipe_2023mar.onnx"))) {
            PoseLandmarks landmarks = detector.detect(frame);
            assertNotNull(landmarks);
            float[] features = landmarks.toFeatureVector();
            assertEquals(66, features.length);
            // The first two values are the nose relative to itself, so they are zero.
            assertEquals(0f, features[0], 1e-6f);
            assertEquals(0f, features[1], 1e-6f);
        }
    }

    private static Mat readFrame() {
        return org.opencv.imgcodecs.Imgcodecs.imread(resource("pose.jpg").toString());
    }

    /** Reads the expected landmark pixel coordinates from the JSON fixture. */
    private static double[][] readExpectedLandmarks() throws Exception {
        String json;
        try (InputStream in = Files.newInputStream(resource("expected_landmarks.json"))) {
            json = new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
        int start = json.indexOf("\"landmarks\"");
        int open = json.indexOf('[', start);
        int depth = 0;
        int end = open;
        for (int i = open; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '[') {
                depth++;
            } else if (c == ']') {
                depth--;
                if (depth == 0) {
                    end = i;
                    break;
                }
            }
        }
        String block = json.substring(open, end + 1);

        java.util.List<double[]> values = new java.util.ArrayList<>();
        for (String row : block.split("\\],\\s*\\[")) {
            String cleaned = row.replace("[", "").replace("]", "").trim();
            if (cleaned.isEmpty()) {
                continue;
            }
            String[] parts = cleaned.split(",");
            values.add(new double[]{
                    Double.parseDouble(parts[0].trim()),
                    Double.parseDouble(parts[1].trim())
            });
        }
        return values.toArray(new double[0][]);
    }
}
