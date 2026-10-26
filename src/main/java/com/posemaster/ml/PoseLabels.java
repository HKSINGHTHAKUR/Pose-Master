package com.posemaster.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Reads the label list that accompanies the trained pose-classification model.
 *
 * <p>The labels are stored in {@code labels.txt}, one UTF-8 label per line, in the order of the
 * classifier's output classes.  Blank lines are ignored.
 */
public final class PoseLabels {

    private PoseLabels() {
    }

    /** Loads the labels from a stream positioned at the start of a {@code labels.txt} file. */
    public static List<String> load(InputStream in) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        List<String> labels = reader.lines().map(String::strip).filter(line -> !line.isEmpty()).toList();
        if (labels.isEmpty()) {
            throw new IOException("Label file is empty");
        }
        return labels;
    }
}
