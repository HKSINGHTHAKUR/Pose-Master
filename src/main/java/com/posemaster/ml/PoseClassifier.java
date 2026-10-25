package com.posemaster.ml;

import java.util.List;

/**
 * Maps a pose feature vector to one of the trained pose labels.
 */
public interface PoseClassifier extends AutoCloseable {

    /**
     * Returns the label with the highest predicted probability.
     *
     * @param features 66 values: for each of the 33 landmarks, {@code (x - x0, y - y0)} where
     *                 landmark 0 is the nose and the coordinates are normalised image coordinates.
     */
    String classify(float[] features);

    /** Returns the full probability vector, one entry per label. */
    float[] predict(float[] features);

    /** Returns the label names in the same order as the model's outputs. */
    List<String> labels();

    @Override
    void close();
}
