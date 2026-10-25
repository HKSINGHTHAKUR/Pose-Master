package com.posemaster.tools;

import com.posemaster.PoseMasterApp;
import com.posemaster.app.HighScoreStore;
import com.posemaster.app.ResourceLocator;
import com.posemaster.camera.WebcamCapture;
import com.posemaster.ml.OnnxPoseClassifier;
import com.posemaster.ml.PoseClassifier;
import com.posemaster.pose.MediapipePoseDetector;
import com.posemaster.pose.PoseDetector;
import com.posemaster.pose.PoseLandmarks;
import com.posemaster.ui.Assets;
import com.posemaster.ui.GameFrame;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Headless smoke test: loads every model and asset, runs the full pose pipeline on a still
 * image, renders the game's screens to PNG files and reports the results.
 *
 * <p>It has no {@code @Test} methods, so Maven does not run it; launch {@code main} from an IDE
 * with the test classpath, optionally passing {@code [image] [outputDir]}.
 */
public final class SmokeTest {

    private SmokeTest() {
    }

    public static void main(String[] args) throws Exception {
        OpenCV.loadLocally();

        ResourceLocator resources = new ResourceLocator();
        System.out.println("resource root: " + resources.root());

        Path classifierModel = resources.resolve(PoseMasterApp.CLASSIFIER_MODEL);
        Path detectorModel = resources.resolve(PoseMasterApp.PERSON_DETECTOR);
        Path landmarkModel = resources.resolve(PoseMasterApp.POSE_LANDMARKER);
        System.out.println("classifier: " + classifierModel);
        System.out.println("detector  : " + detectorModel);
        System.out.println("landmarker: " + landmarkModel);

        PoseClassifier classifier = OnnxPoseClassifier.create(classifierModel,
                resources.resolve(PoseMasterApp.LABELS));
        System.out.println("labels    : " + classifier.labels().size() + " -> " + classifier.labels());

        PoseDetector detector = MediapipePoseDetector.create(detectorModel, landmarkModel);

        // Exercise the pose pipeline on a still image.
        Path image = Paths.get(args.length > 0 ? args[0] : "src/test/resources/pose.jpg");
        Mat frame = Imgcodecs.imread(image.toString());
        if (frame == null || frame.empty()) {
            System.out.println("SMOKE: could not read " + image);
        } else {
            PoseLandmarks landmarks = detector.detect(frame);
            if (landmarks == null) {
                System.out.println("SMOKE: no pose detected in " + image);
            } else {
                float[] features = landmarks.toFeatureVector();
                String pose = classifier.classify(features);
                System.out.println("SMOKE: pose detected, body fully visible = "
                        + landmarks.isBodyFullyVisible() + ", predicted pose = '" + pose + "'");
                System.out.println("SMOKE: feature vector length = " + features.length);
            }
        }

        // Load every asset the game needs.
        Assets assets = new Assets(resources);
        int imageCount = 0;
        for (String name : new String[]{
                "Background_Images/Background1.png", "Background_Images/Background2.png",
                "Background_Images/Background3.png", "Background_Images/Background4.png",
                "Images/title.png", "Images/title_with_man.png", "Images/clock.png",
                "Images/not_visible.png", "Images/gameover3.png"}) {
            BufferedImage loaded = assets.image(name);
            if (loaded == null) {
                throw new IllegalStateException("missing asset " + name);
            }
            imageCount++;
        }
        for (String label : classifier.labels()) {
            BufferedImage loaded = assets.image("poseImages/" + label + ".png");
            if (loaded == null) {
                throw new IllegalStateException("missing pose image " + label);
            }
            imageCount++;
        }
        for (String font : new String[]{
                Assets.FONT_MARCELLUS, Assets.FONT_SILKSCREEN, Assets.FONT_NABLA, Assets.FONT_GOD_OF_WAR}) {
            if (assets.font(font, 24) == null) {
                throw new IllegalStateException("missing font " + font);
            }
        }
        System.out.println("SMOKE: loaded " + imageCount + " images and 4 fonts");

        // Build the window and render each screen.
        File outputDir = new File(args.length > 1 ? args[1] : "target/smoke");
        outputDir.mkdirs();

        WebcamCapture camera = new WebcamCapture();
        HighScoreStore store = new HighScoreStore(Paths.get(outputDir.getAbsolutePath(), "high_score.txt"));
        GameFrame frame2 = new GameFrame(assets, classifier, detector, camera, store, classifier.labels());

        SwingUtilities.invokeAndWait(() -> {
            for (String screen : new String[]{"home", "start", "game", "gameover"}) {
                frame2.setScreenForTest(screen);
                frame2.setSize(frame2.getWidth(), frame2.getHeight());
                frame2.validate();
                BufferedImage shot = new BufferedImage(GameFrame.WIDTH, GameFrame.HEIGHT,
                        BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D g = shot.createGraphics();
                frame2.paint(g);
                g.dispose();
                try {
                    File out = new File(outputDir, screen + ".png");
                    ImageIO.write(shot, "png", out);
                    System.out.println("SMOKE: rendered " + out.getAbsolutePath() + " ("
                            + out.length() + " bytes)");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        classifier.close();
        detector.close();
        camera.close();
        System.out.println("SMOKE: OK");
    }
}
