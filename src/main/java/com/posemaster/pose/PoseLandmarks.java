package com.posemaster.pose;

import java.util.List;

/**
 * The set of body landmarks produced by the pose detector, in MediaPipe's 33-point BlazePose
 * ordering.
 *
 * <p>The trained classifier was fitted on this exact layout, so the index constants below are
 * part of the model contract and must not be renumbered.
 */
public final class PoseLandmarks {

    /** Number of landmarks in the MediaPipe pose topology. */
    public static final int LANDMARK_COUNT = 33;

    public static final int NOSE = 0;
    public static final int LEFT_EYE_INNER = 1;
    public static final int LEFT_EYE = 2;
    public static final int LEFT_EYE_OUTER = 3;
    public static final int RIGHT_EYE_INNER = 4;
    public static final int RIGHT_EYE = 5;
    public static final int RIGHT_EYE_OUTER = 6;
    public static final int LEFT_EAR = 7;
    public static final int RIGHT_EAR = 8;
    public static final int MOUTH_LEFT = 9;
    public static final int MOUTH_RIGHT = 10;
    public static final int LEFT_SHOULDER = 11;
    public static final int RIGHT_SHOULDER = 12;
    public static final int LEFT_ELBOW = 13;
    public static final int RIGHT_ELBOW = 14;
    public static final int LEFT_WRIST = 15;
    public static final int RIGHT_WRIST = 16;
    public static final int LEFT_PINKY = 17;
    public static final int RIGHT_PINKY = 18;
    public static final int LEFT_INDEX = 19;
    public static final int RIGHT_INDEX = 20;
    public static final int LEFT_THUMB = 21;
    public static final int RIGHT_THUMB = 22;
    public static final int LEFT_HIP = 23;
    public static final int RIGHT_HIP = 24;
    public static final int LEFT_KNEE = 25;
    public static final int RIGHT_KNEE = 26;
    public static final int LEFT_ANKLE = 27;
    public static final int RIGHT_ANKLE = 28;
    public static final int LEFT_HEEL = 29;
    public static final int RIGHT_HEEL = 30;
    public static final int LEFT_FOOT_INDEX = 31;
    public static final int RIGHT_FOOT_INDEX = 32;

    /**
     * The landmarks whose visibility the original game required to be above threshold before it
     * would classify a pose: both ankles and both wrists.
     */
    public static final int[] REQUIRED_VISIBLE_LANDMARKS = {
            RIGHT_ANKLE, LEFT_ANKLE, LEFT_WRIST, RIGHT_WRIST
    };

    /** Minimum visibility the original game required for the landmarks above. */
    public static final float VISIBILITY_THRESHOLD = 0.6f;

    private final List<PoseLandmark> landmarks;

    public PoseLandmarks(List<PoseLandmark> landmarks) {
        if (landmarks.size() != LANDMARK_COUNT) {
            throw new IllegalArgumentException(
                    "Expected " + LANDMARK_COUNT + " landmarks, got " + landmarks.size());
        }
        this.landmarks = List.copyOf(landmarks);
    }

    public List<PoseLandmark> all() {
        return landmarks;
    }

    public PoseLandmark get(int index) {
        return landmarks.get(index);
    }

    /**
     * Mirrors the original game's "is the player fully in frame" test: the two ankles and the two
     * wrists must all be detected with sufficient confidence.
     */
    public boolean isBodyFullyVisible() {
        for (int index : REQUIRED_VISIBLE_LANDMARKS) {
            if (landmarks.get(index).visibility() < VISIBILITY_THRESHOLD) {
                return false;
            }
        }
        return true;
    }

    /**
     * Builds the 66-value feature vector the classifier expects: for each landmark, its
     * coordinates relative to the nose (landmark 0), in the original {@code x, y} interleaved
     * order.
     */
    public float[] toFeatureVector() {
        PoseLandmark nose = landmarks.get(NOSE);
        float[] features = new float[LANDMARK_COUNT * 2];
        for (int i = 0; i < LANDMARK_COUNT; i++) {
            PoseLandmark landmark = landmarks.get(i);
            features[i * 2] = landmark.x() - nose.x();
            features[i * 2 + 1] = landmark.y() - nose.y();
        }
        return features;
    }
}
