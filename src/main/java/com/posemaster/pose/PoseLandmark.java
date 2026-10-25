package com.posemaster.pose;

/**
 * A single body landmark in normalised image coordinates.
 *
 * @param x         horizontal position, normalised by the image width (0..1)
 * @param y         vertical position, normalised by the image height (0..1)
 * @param visibility confidence that the landmark is present and unoccluded
 */
public record PoseLandmark(float x, float y, float visibility) {

    /** Returns a copy of this landmark with the given visibility. */
    public PoseLandmark withVisibility(float newVisibility) {
        return new PoseLandmark(x, y, newVisibility);
    }
}
