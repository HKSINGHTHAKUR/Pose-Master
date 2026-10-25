package com.posemaster.game;

import java.util.List;
import java.util.Random;

/**
 * The rules of a single game session, independent of any UI.
 *
 * <p>Mirrors the original Python implementation:
 *
 * <ul>
 *   <li>a session lasts 60 seconds, and every matched pose adds 1.5 seconds;</li>
 *   <li>the target pose changes every 7 seconds, and immediately after a match;</li>
 *   <li>a new target pose is always different from the one it replaces.</li>
 * </ul>
 *
 * <p>All timing is driven by the caller passing the elapsed time, so the class is fully
 * testable without a camera or a UI.
 */
public final class GameSession {

    /** Base length of a game session in seconds. */
    public static final double GAME_DURATION_SECONDS = 60.0;
    /** Seconds added to the timer for each matched pose. */
    public static final double TIME_BONUS_SECONDS = 1.5;
    /** Seconds a target pose stays on screen before it is replaced. */
    public static final double POSE_DURATION_SECONDS = 7.0;

    private final List<String> labels;
    private final Random random;

    private double startTime;
    private String targetPose;
    private String previousPose = "";
    private double levelStartTime;
    private double remainingSeconds = GAME_DURATION_SECONDS;
    private int score;
    private boolean levelCompleted;
    private boolean finished;
    private boolean started;

    public GameSession(List<String> labels, Random random) {
        if (labels == null || labels.isEmpty()) {
            throw new IllegalArgumentException("At least one pose label is required");
        }
        this.labels = List.copyOf(labels);
        this.random = random;
        this.targetPose = pickRandomPose();
    }

    /** Starts the session clock. */
    public void start(double now) {
        this.startTime = now;
        this.levelStartTime = now;
        this.remainingSeconds = GAME_DURATION_SECONDS;
        this.score = 0;
        this.finished = false;
        this.started = true;
        this.levelCompleted = false;
        this.previousPose = "";
        this.targetPose = pickRandomPose();
    }

    /** Returns the pose the player currently has to match. */
    public String targetPose() {
        return targetPose;
    }

    /** Returns the number of matched poses so far. */
    public int score() {
        return score;
    }

    /** Returns the seconds left on the clock. */
    public double remainingSeconds() {
        return remainingSeconds;
    }

    /** Returns the whole seconds left on the clock, as shown on the in-game timer. */
    public int remainingWholeSeconds() {
        return (int) remainingSeconds;
    }

    /** Returns how long the current target pose has been shown, in seconds. */
    public double poseElapsedSeconds(double now) {
        return now - levelStartTime;
    }

    /** Returns whether the session has run out of time. */
    public boolean isFinished() {
        return finished;
    }

    /** Returns the total time the session lasted, in seconds. */
    public double elapsedSeconds(double now) {
        return now - startTime;
    }

    /**
     * Advances the session by one frame.
     *
     * @param now           seconds since the session started
     * @param predictedPose the pose the classifier currently reports, or {@code null} if none
     * @param bodyVisible   whether the player's whole body was detected
     */
    public void update(double now, String predictedPose, boolean bodyVisible) {
        if (!started || finished) {
            return;
        }

        // The target pose expires on its own after seven seconds.
        if (now - levelStartTime >= POSE_DURATION_SECONDS) {
            levelCompleted = true;
            levelStartTime = now;
        }

        if (levelCompleted) {
            chooseNewPose();
        }

        if (bodyVisible && predictedPose != null && predictedPose.equals(targetPose)) {
            score++;
            levelStartTime = now;
            // The pose changes on the next update, as in the original.
            levelCompleted = true;
        }

        // The total time grows with the score: 60 seconds plus 1.5 seconds per matched pose.
        double totalTime = startTime + GAME_DURATION_SECONDS + TIME_BONUS_SECONDS * score;
        if (now >= totalTime) {
            remainingSeconds = 0;
            finished = true;
            return;
        }
        remainingSeconds = totalTime - now;
    }

    private void chooseNewPose() {
        previousPose = targetPose;
        do {
            targetPose = pickRandomPose();
        } while (targetPose.equals(previousPose));
        levelCompleted = false;
    }

    private String pickRandomPose() {
        return labels.get(random.nextInt(labels.size()));
    }
}
