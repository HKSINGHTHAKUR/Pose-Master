package com.posemaster.pose;

import org.opencv.core.Mat;

/**
 * Detects body landmarks in a camera frame.
 *
 * <p>Implementations return {@code null} when no pose could be found in the frame.
 */
public interface PoseDetector extends AutoCloseable {

    /**
     * Detects pose landmarks in the given BGR frame.
     *
     * @param frame BGR image as produced by OpenCV
     * @return the detected landmarks, or {@code null} if no pose was found
     */
    PoseLandmarks detect(Mat frame);

    @Override
    void close();
}
