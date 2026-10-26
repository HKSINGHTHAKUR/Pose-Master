package com.posemaster.game;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GameSessionTest {

    private static final List<String> LABELS = List.of("alpha", "beta", "gamma");

    private static GameSession newSession(long seed) {
        GameSession session = new GameSession(LABELS, new Random(seed));
        session.start(0);
        return session;
    }

    @Test
    void startsWithSixtySecondsAndNoScore() {
        GameSession session = newSession(1);
        assertEquals(60.0, session.remainingSeconds(), 1e-9);
        assertEquals(0, session.score());
        assertFalse(session.isFinished());
        assertTrue(LABELS.contains(session.targetPose()));
    }

    @Test
    void matchingTheTargetPoseScoresAndAddsTime() {
        GameSession session = newSession(2);
        String target = session.targetPose();

        session.update(1.0, target, true);

        assertEquals(1, session.score());
        // One second elapsed, plus the 1.5 second bonus from the match.
        assertEquals(60.0 + 1.5 - 1.0, session.remainingSeconds(), 1e-9);
    }

    @Test
    void aWrongPoseDoesNotScore() {
        GameSession session = newSession(3);
        String wrong = LABELS.stream().filter(l -> !l.equals(session.targetPose())).findFirst().orElseThrow();

        session.update(1.0, wrong, true);

        assertEquals(0, session.score());
    }

    @Test
    void anInvisibleBodyDoesNotScoreEvenOnAMatch() {
        GameSession session = newSession(4);
        String target = session.targetPose();

        session.update(1.0, target, false);

        assertEquals(0, session.score());
    }

    @Test
    void theTargetPoseChangesAfterSevenSeconds() {
        GameSession session = newSession(5);
        String first = session.targetPose();

        // Advance to just before the pose would expire.
        session.update(6.9, null, false);
        assertEquals(first, session.targetPose());

        session.update(7.1, null, false);
        assertNotEquals(first, session.targetPose());
    }

    @Test
    void theTargetPoseChangesAfterAMatch() {
        GameSession session = newSession(6);
        String first = session.targetPose();

        session.update(1.0, first, true);
        // The change takes effect on the next update.
        assertEquals(first, session.targetPose());

        session.update(1.5, null, false);
        assertNotEquals(first, session.targetPose());
    }

    @Test
    void theSessionEndsWhenTheTimerRunsOut() {
        GameSession session = newSession(7);
        session.update(59.0, null, false);
        assertFalse(session.isFinished());

        session.update(60.1, null, false);
        assertTrue(session.isFinished());
        assertEquals(0.0, session.remainingSeconds(), 1e-9);
    }

    @Test
    void matchingExtendsTheSessionPastSixtySeconds() {
        GameSession session = newSession(8);
        String target = session.targetPose();

        // Score early on, which adds 1.5 seconds to the total time.
        session.update(1.0, target, true);
        session.update(61.0, null, false);

        assertFalse(session.isFinished());
        assertTrue(session.remainingSeconds() > 0);
    }

    @Test
    void theNewTargetIsAlwaysDifferentFromTheOldOne() {
        GameSession session = newSession(9);
        double now = 0;
        // Eight rounds keep the session alive: each round spends 7 seconds but scores a point,
        // which only adds 1.5 seconds to the 60 second timer.
        for (int i = 0; i < 8; i++) {
            session.update(now, session.targetPose(), true);
            assertEquals(i + 1, session.score());

            String before = session.targetPose();
            now += 7.0;
            session.update(now, null, false);
            assertNotEquals(before, session.targetPose());
        }
    }
}
