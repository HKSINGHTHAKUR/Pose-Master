package com.posemaster.pose;

/**
 * Generates the 2254 SSD anchors used by the MediaPipe BlazePose person detector.
 *
 * <p>The detector is an SSD-style model with five layers (strides 8, 16, 32, 32, 32) over a
 * 224x224 input.  Two anchors are emitted per grid position; the three stride-32 layers share
 * a single 7x7 grid, so their anchors coincide and appear as a run of six.
 *
 * <p>Positions are laid out row-major: the y coordinate advances slowest.
 */
final class BlazePoseAnchors {

    /** Number of anchors the person detector produces. */
    static final int COUNT = 2254;

    private static final int DETECTOR_INPUT_SIZE = 224;

    private BlazePoseAnchors() {
    }

    /**
     * Builds the anchor table.
     *
     * @return a flat array of 4508 values, {@code x0, y0, x1, y1, ...}
     */
    static float[] generate() {
        float[] anchors = new float[COUNT * 2];
        int index = 0;

        for (int stride : new int[]{8, 16}) {
            int cells = DETECTOR_INPUT_SIZE / stride;
            for (int y = 0; y < cells; y++) {
                for (int x = 0; x < cells; x++) {
                    float cx = (x + 0.5f) * stride / DETECTOR_INPUT_SIZE;
                    float cy = (y + 0.5f) * stride / DETECTOR_INPUT_SIZE;
                    for (int copy = 0; copy < 2; copy++) {
                        anchors[index++] = cx;
                        anchors[index++] = cy;
                    }
                }
            }
        }

        int cells = DETECTOR_INPUT_SIZE / 32;
        for (int y = 0; y < cells; y++) {
            for (int x = 0; x < cells; x++) {
                float cx = (x + 0.5f) * 32 / DETECTOR_INPUT_SIZE;
                float cy = (y + 0.5f) * 32 / DETECTOR_INPUT_SIZE;
                for (int copy = 0; copy < 6; copy++) {
                    anchors[index++] = cx;
                    anchors[index++] = cy;
                }
            }
        }

        if (index != anchors.length) {
            throw new IllegalStateException("Generated " + index + " anchor values, expected " + anchors.length);
        }
        return anchors;
    }
}
