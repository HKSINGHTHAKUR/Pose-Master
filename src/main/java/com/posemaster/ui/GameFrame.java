package com.posemaster.ui;

import com.posemaster.app.HighScoreStore;
import com.posemaster.camera.WebcamCapture;
import com.posemaster.game.GameSession;
import com.posemaster.ml.OnnxPoseClassifier;
import com.posemaster.ml.PoseClassifier;
import com.posemaster.pose.MediapipePoseDetector;
import com.posemaster.pose.PoseDetector;
import com.posemaster.pose.PoseLandmarks;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The game's Swing window, showing the same four screens as the Python (pygame) version.
 *
 * <p>Camera capture and inference run on a background thread so the event dispatch thread is
 * never blocked; the UI reads the most recent result and repaints at the target frame rate.
 */
public final class GameFrame extends JFrame {

    /** Window size, matching the Python (pygame) window. */
    public static final int WIDTH = 1240;
    public static final int HEIGHT = 700;
    /** Target frame rate, matching the Python version. */
    public static final int FPS = 30;

    /** The screens the game moves between. */
    private enum Screen {
        HOME, START, GAME, GAME_OVER
    }

    private final Assets assets;
    private final PoseClassifier classifier;
    private final PoseDetector detector;
    private final WebcamCapture camera;
    private final HighScoreStore highScoreStore;

    private final AtomicReference<BufferedImage> preview = new AtomicReference<>();
    private final AtomicReference<PoseLandmarks> latestLandmarks = new AtomicReference<>();
    private final AtomicBoolean detectionFailed = new AtomicBoolean();

    private volatile Screen screen = Screen.HOME;
    private GameSession session;
    private int highScore;
    private int lastScore;
    private String lastPredictedPose;
    private long lastFrameNanos;
    private double gameStartNanos;
    private double gameOverTime;
    private long gameOverNanos;

    private Font marcellus;
    private Font silkscreen;
    private Font godOfWar;
    private Font nabla;

    private final Thread visionThread;
    private volatile boolean running = true;

    public GameFrame(Assets assets, PoseClassifier classifier, PoseDetector detector,
                     WebcamCapture camera, HighScoreStore highScoreStore, List<String> labels) {
        super("Pose Master");
        this.assets = assets;
        this.classifier = classifier;
        this.detector = detector;
        this.camera = camera;
        this.highScoreStore = highScoreStore;
        this.session = new GameSession(labels, new Random());

        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setLocationRelativeTo(null);
        setResizable(false);

        try {
            marcellus = assets.font(Assets.FONT_MARCELLUS, 60);
            silkscreen = assets.font(Assets.FONT_SILKSCREEN, 40);
            godOfWar = assets.font(Assets.FONT_GOD_OF_WAR, 40);
            nabla = assets.font(Assets.FONT_NABLA, 40);
        } catch (IOException e) {
            throw new IllegalStateException("Could not load the bundled fonts", e);
        }

        this.highScore = highScoreStore.load(0);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                handleClick(e.getX(), e.getY());
            }
        });

        visionThread = new Thread(this::visionLoop, "pose-master-vision");
        visionThread.setDaemon(true);
    }

    /** Starts the background capture and inference thread. */
    public void start() {
        visionThread.start();
    }

    /** Stops the background thread and releases the camera. */
    public void shutdown() {
        running = false;
        visionThread.interrupt();
        camera.close();
    }

    // ---------------------------------------------------------------- background work

    private void visionLoop() {
        lastFrameNanos = System.nanoTime();
        while (running) {
            long frameStart = System.nanoTime();
            try {
                Mat frame = camera.read();
                if (frame == null) {
                    detectionFailed.set(true);
                    sleepUntilNextFrame(frameStart);
                    continue;
                }
                detectionFailed.set(false);

                // The Python version mirrored the camera image so the player sees themselves the
                // right way round; the pose model is fed the same mirrored frame.
                Mat flipped = new Mat();
                Core.flip(frame, flipped, 1);

                BufferedImage image = toBufferedImage(flipped);
                preview.set(image);

                PoseLandmarks landmarks = detector.detect(flipped);
                latestLandmarks.set(landmarks);

                String predicted = null;
                boolean bodyVisible = landmarks != null && landmarks.isBodyFullyVisible();
                if (bodyVisible) {
                    predicted = classifier.classify(landmarks.toFeatureVector());
                }
                lastPredictedPose = predicted;

                if (screen == Screen.GAME) {
                    session.update(elapsedSeconds(), predicted, bodyVisible);
                    if (session.isFinished()) {
                        endGame();
                    }
                }
                flipped.release();
                frame.release();
            } catch (RuntimeException e) {
                detectionFailed.set(true);
            }
            sleepUntilNextFrame(frameStart);
        }
    }

    private void sleepUntilNextFrame(long frameStart) {
        long frameNanos = 1_000_000_000L / FPS;
        long elapsed = System.nanoTime() - frameStart;
        long sleep = frameNanos - elapsed;
        if (sleep > 0) {
            try {
                Thread.sleep(sleep / 1_000_000L, (int) (sleep % 1_000_000L));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private static BufferedImage toBufferedImage(Mat mat) {
        int width = mat.cols();
        int height = mat.rows();
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        byte[] data = new byte[width * height * 3];
        mat.get(0, 0, data);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src = (y * width + x) * 3;
                int dst = (y * width + x) * 3;
                image.getRaster().setDataElements(x, y, new byte[]{
                        data[src + 2], data[src + 1], data[src]
                });
            }
        }
        return image;
    }

    // ---------------------------------------------------------------- input

    private void handleClick(int x, int y) {
        switch (screen) {
            case HOME -> {
                if (homeStartButton().contains(x, y)) {
                    screen = Screen.START;
                    repaint();
                } else if (homeQuitButton().contains(x, y)) {
                    dispose();
                }
            }
            case START -> {
                if (startScreenStartButton().contains(x, y)) {
                    beginGame();
                } else if (startScreenBackButton().contains(x, y)) {
                    screen = Screen.HOME;
                    repaint();
                }
            }
            case GAME -> {
                if (gameQuitButton().contains(x, y)) {
                    endGame();
                }
            }
            case GAME_OVER -> {
                if (gameOverPlayAgainButton().contains(x, y)) {
                    beginGame();
                } else if (gameOverMenuButton().contains(x, y)) {
                    screen = Screen.HOME;
                    repaint();
                }
            }
            default -> {
            }
        }
    }

    // The button geometry matches the Python (pygame) layout, so the hit boxes and the drawn
    // buttons can never drift apart.
    private Button homeStartButton() {
        return new Button(270, 500, 300, 100, Color.BLACK, Color.WHITE, Color.WHITE, marcellus, "Start");
    }

    private Button homeQuitButton() {
        return new Button(670, 500, 300, 100, Color.BLACK, Color.WHITE, Color.WHITE, marcellus, "Quit");
    }

    private Button startScreenStartButton() {
        return new Button(1030, 200, 200, 70, new Color(0x34eb5b), Color.BLACK, Color.BLACK,
                marcellus.deriveFont(50f), "Start");
    }

    private Button startScreenBackButton() {
        return new Button(1030, 300, 200, 70, new Color(255, 0, 0), Color.BLACK, Color.BLACK,
                marcellus.deriveFont(50f), "Back");
    }

    private Button gameQuitButton() {
        return new Button(1030, 300, 200, 70, new Color(255, 0, 0), Color.BLACK, Color.BLACK,
                marcellus.deriveFont(50f), "QUIT");
    }

    private Button gameOverPlayAgainButton() {
        return new Button(220, 550, 430, 100, new Color(0x49e68a), Color.BLACK, Color.BLACK,
                marcellus.deriveFont(60f), "Play Again");
    }

    private Button gameOverMenuButton() {
        return new Button(620, 550, 430, 100, new Color(0xe32938), Color.BLACK, Color.BLACK,
                marcellus.deriveFont(60f), "Main Menu");
    }

    private void beginGame() {
        session = new GameSession(classifier.labels(), new Random());
        gameStartNanos = System.nanoTime();
        session.start(0);
        lastScore = 0;
        screen = Screen.GAME;
        repaint();
    }

    /**
     * Switches to a screen by name.  Used by the headless smoke test to render every screen;
     * the game itself only ever moves between screens through the buttons.
     */
    public void setScreenForTest(String name) {
        switch (name) {
            case "home" -> screen = Screen.HOME;
            case "start" -> screen = Screen.START;
            case "game" -> {
                session = new GameSession(classifier.labels(), new Random());
                session.start(0);
                gameStartNanos = System.nanoTime();
                screen = Screen.GAME;
            }
            case "gameover" -> {
                session = new GameSession(classifier.labels(), new Random());
                session.start(0);
                lastScore = 5;
                gameOverTime = 42.5;
                gameOverNanos = System.nanoTime() - 2_000_000_000L;
                screen = Screen.GAME_OVER;
            }
            default -> throw new IllegalArgumentException("Unknown screen: " + name);
        }
        repaint();
    }

    private void endGame() {
        lastScore = session.score();
        if (lastScore > highScore) {
            highScore = lastScore;
            try {
                highScoreStore.save(highScore);
            } catch (IOException e) {
                // A failed write only costs the player their high score.
            }
        }
        gameOverTime = session.elapsedSeconds(elapsedSeconds());
        gameOverNanos = System.nanoTime();
        screen = Screen.GAME_OVER;
        repaint();
    }

    private double elapsedSeconds() {
        return (System.nanoTime() - gameStartNanos) / 1_000_000_000.0;
    }

    // ---------------------------------------------------------------- painting

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        try {
            switch (screen) {
                case HOME -> paintHome(g2);
                case START -> paintStart(g2);
                case GAME -> paintGame(g2);
                case GAME_OVER -> paintGameOver(g2);
                default -> {
                }
            }
        } catch (IOException e) {
            g2.setColor(Color.RED);
            g2.drawString("Could not load a game asset: " + e.getMessage(), 20, 30);
        }
        g2.dispose();
    }

    private void paintHome(Graphics2D g) throws IOException {
        drawScaled(g, "Background_Images/Background4.png", WIDTH, HEIGHT);
        drawScaled(g, "Images/title_with_man.png", 800, 800, 200, -150);
        homeStartButton().draw(g);
        homeQuitButton().draw(g);
    }

    private void paintStart(Graphics2D g) throws IOException {
        drawScaled(g, "Background_Images/Background1.png", WIDTH, HEIGHT);
        new TextOverlay(10, 10, godOfWar.deriveFont(35f), new Color(0x1c1e38), "HighScore").draw(g);
        new TextOverlay(10, 50, godOfWar.deriveFont(50f), new Color(0x1c1e38),
                Integer.toString(highScore)).draw(g);

        drawCameraPreview(g);
        drawScaled(g, "Images/title.png", 600, 600, 335, 335);

        startScreenStartButton().draw(g);
        startScreenBackButton().draw(g);
    }

    private void paintGame(Graphics2D g) throws IOException {
        drawScaled(g, "Background_Images/Background1.png", WIDTH, HEIGHT);

        new TextOverlay(1040, 5, silkscreen.deriveFont(50f), new Color(0xf2e30f), "Score:").draw(g);
        new TextOverlay(1150, 50, silkscreen.deriveFont(60f), new Color(0xf2e30f),
                Integer.toString(session.score())).draw(g);
        new TextOverlay(150, 570, godOfWar.deriveFont(70f), Color.WHITE,
                session.targetPose()).draw(g);

        drawScaled(g, "poseImages/" + session.targetPose() + ".png", 370, 250, 800, 500);

        drawScaled(g, "Images/clock.png", 70, 70, 10, 10);
        new TextOverlay(100, 10, silkscreen.deriveFont(50f), new Color(0x1c1e38),
                Integer.toString(session.remainingWholeSeconds())).draw(g);

        drawCameraPreview(g);

        gameQuitButton().draw(g);

        PoseLandmarks landmarks = latestLandmarks.get();
        if (landmarks == null || !landmarks.isBodyFullyVisible()) {
            new TextOverlay(140, 450, silkscreen.deriveFont(40f), new Color(255, 0, 0),
                    "! Make Sure your whole body is visible").draw(g);
            drawScaled(g, "Images/not_visible.png", 220, 220, 0, 230);
        }
    }

    private void paintGameOver(Graphics2D g) throws IOException {
        drawScaled(g, "Background_Images/Background2.png", WIDTH, HEIGHT);

        double sinceGameOver = (System.nanoTime() - gameOverNanos) / 1_000_000_000.0;
        if (sinceGameOver < 1.5) {
            drawScaled(g, "Images/gameover3.png", 600, 400, 300, 100);
            return;
        }

        drawScaled(g, "Images/title.png", 800, 800, 250, -200);
        new TextOverlay(100, 300, silkscreen.deriveFont(50f), new Color(0x252945),
                "Score: " + lastScore).draw(g);
        new TextOverlay(600, 300, silkscreen.deriveFont(50f), new Color(0x252945),
                "Time Played: " + (int) gameOverTime + " s ").draw(g);
        new TextOverlay(100, 400, silkscreen.deriveFont(50f), new Color(0x252945),
                "HighScore: " + highScore).draw(g);
        new TextOverlay(600, 400, silkscreen.deriveFont(50f), new Color(0x252945),
                String.format("Time / Move: %s s", formatTimePerMove())).draw(g);

        gameOverMenuButton().draw(g);
        gameOverPlayAgainButton().draw(g);
    }

    /** Matches the Python version's "time per move" figure. */
    private String formatTimePerMove() {
        if (lastScore == 0) {
            return Integer.toString((int) gameOverTime);
        }
        double perMove = gameOverTime / (lastScore + 1);
        return String.format("%.2f", perMove);
    }

    private void drawCameraPreview(Graphics2D g) {
        BufferedImage frame = preview.get();
        if (frame == null) {
            if (detectionFailed.get()) {
                new TextOverlay(220, 300, silkscreen.deriveFont(30f), Color.RED,
                        "No webcam detected - start Pose Master with a camera connected")
                        .draw(g);
            }
            return;
        }
        // The Python version blitted the camera frame at its native size, at (220, 0).
        g.drawImage(frame, 220, 0, null);
    }

    private void drawScaled(Graphics2D g, String name, int width, int height) throws IOException {
        drawScaled(g, name, width, height, 0, 0);
    }

    private void drawScaled(Graphics2D g, String name, int width, int height, int x, int y)
            throws IOException {
        BufferedImage image = assets.image(name);
        g.drawImage(image, x, y, width, height, null);
    }
}
