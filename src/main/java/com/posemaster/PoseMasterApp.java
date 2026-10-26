package com.posemaster;

import com.posemaster.app.HighScoreStore;
import com.posemaster.app.ResourceLocator;
import com.posemaster.camera.WebcamCapture;
import com.posemaster.ml.OnnxPoseClassifier;
import com.posemaster.ml.PoseClassifier;
import com.posemaster.pose.MediapipePoseDetector;
import com.posemaster.pose.PoseDetector;
import com.posemaster.ui.Assets;
import com.posemaster.ui.GameFrame;
import nu.pattern.OpenCV;
import org.opencv.core.Core;

import javax.swing.SwingUtilities;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;

/**
 * Entry point of Pose Master.
 *
 * <p>The application loads the trained pose-classification model and the MediaPipe pose models,
 * opens the webcam and shows the game window.
 */
public final class PoseMasterApp {

    /** Pose classifier model. */
    public static final String CLASSIFIER_MODEL = "models/pose_classifier.onnx";
    /** Pose label list, {@code labels.txt}. */
    public static final String LABELS = "labels.txt";
    /** MediaPipe person detector. */
    public static final String PERSON_DETECTOR = "models/person_detection_mediapipe_2023mar.onnx";
    /** MediaPipe pose landmark model. */
    public static final String POSE_LANDMARKER = "models/pose_estimation_mediapipe_2023mar.onnx";

    private PoseMasterApp() {
    }

    public static void main(String[] args) {
        ResourceLocator resources = new ResourceLocator();
        try {
            OpenCV.loadLocally();
        } catch (Throwable t) {
            System.err.println("Could not load the OpenCV native library: " + t.getMessage());
        }

        try {
            Path classifierModel = resources.resolve(CLASSIFIER_MODEL);
            Path personDetector = resources.resolve(PERSON_DETECTOR);
            Path poseLandmarker = resources.resolve(POSE_LANDMARKER);

            PoseClassifier classifier = OnnxPoseClassifier.create(classifierModel, resources.resolve(LABELS));
            PoseDetector detector = MediapipePoseDetector.create(personDetector, poseLandmarker);
            List<String> labels = classifier.labels();

            WebcamCapture camera = new WebcamCapture();
            HighScoreStore highScoreStore = new HighScoreStore();
            Assets assets = new Assets(resources);

            SwingUtilities.invokeLater(() -> {
                GameFrame frame = new GameFrame(assets, classifier, detector, camera, highScoreStore, labels);
                frame.addWindowListener(new java.awt.event.WindowAdapter() {
                    @Override
                    public void windowClosed(java.awt.event.WindowEvent e) {
                        frame.shutdown();
                        classifier.close();
                        detector.close();
                    }
                });
                frame.setVisible(true);
                frame.start();

                // Drive the repaints from the event dispatch thread at the game's frame rate.
                javax.swing.Timer timer = new javax.swing.Timer(1000 / GameFrame.FPS, e -> frame.repaint());
                timer.start();
            });
        } catch (IOException | ai.onnxruntime.OrtException e) {
            System.err.println("Could not start Pose Master: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
