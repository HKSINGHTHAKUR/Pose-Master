package com.posemaster.camera;

import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.Locale;

/**
 * Webcam capture built on OpenCV.
 *
 * <p>Mirrors the Python version: it tries a list of platform-appropriate capture
 * backends in turn and falls back to the default backend when none of them open the device.  The
 * capture is opened lazily on the first read so that constructing the object never blocks.
 */
public final class WebcamCapture implements AutoCloseable {

    /** Frame width requested from the camera, matching the Python version. */
    public static final int REQUESTED_WIDTH = 800;
    /** Frame height requested from the camera, matching the Python version. */
    public static final int REQUESTED_HEIGHT = 100;

    private final int cameraIndex;
    private VideoCapture capture;
    private boolean warned;

    public WebcamCapture() {
        this(0);
    }

    public WebcamCapture(int cameraIndex) {
        this.cameraIndex = cameraIndex;
    }

    /**
     * Reads the next frame.
     *
     * @return the frame, or {@code null} if the camera is unavailable or the read failed
     */
    public Mat read() {
        if (capture == null) {
            open();
        }
        if (capture == null || !capture.isOpened()) {
            return null;
        }
        Mat frame = new Mat();
        boolean ok = capture.read(frame);
        if (!ok || frame.empty()) {
            frame.release();
            return null;
        }
        return frame;
    }

    /** Opens the camera, trying each supported backend in turn. */
    public void open() {
        if (capture != null) {
            return;
        }
        for (int backend : candidateBackends()) {
            VideoCapture attempt = new VideoCapture(cameraIndex, backend);
            if (attempt.isOpened()) {
                capture = attempt;
                configure();
                return;
            }
            attempt.release();
        }
        // Last resort: let OpenCV pick the default backend.
        VideoCapture fallback = new VideoCapture(cameraIndex);
        if (fallback.isOpened()) {
            capture = fallback;
            configure();
        } else {
            fallback.release();
        }
    }

    private void configure() {
        capture.set(3, REQUESTED_WIDTH);
        capture.set(4, REQUESTED_HEIGHT);
    }

    /** Returns the capture backends to try, in order, for the current operating system. */
    static int[] candidateBackends() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("win")) {
            return new int[]{
                    Videoio.CAP_DSHOW,
                    Videoio.CAP_MSMF,
                    Videoio.CAP_ANY
            };
        }
        if (os.contains("mac")) {
            return new int[]{
                    Videoio.CAP_AVFOUNDATION,
                    Videoio.CAP_ANY
            };
        }
        return new int[]{
                Videoio.CAP_V4L2,
                Videoio.CAP_ANY
        };
    }

    /** Returns whether the camera has been opened successfully. */
    public boolean isOpen() {
        return capture != null && capture.isOpened();
    }

    /** Returns whether a warning about the unavailable camera has already been shown. */
    public boolean isWarned() {
        return warned;
    }

    /** Marks the camera warning as shown, so it is only reported once. */
    public void markWarned() {
        warned = true;
    }

    @Override
    public void close() {
        if (capture != null) {
            capture.release();
            capture = null;
        }
    }
}
