package com.posemaster.ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;

/**
 * Draws a piece of text at a fixed position, the way the Python version's {@code Text} helper did.
 */
public final class TextOverlay {

    private final int x;
    private final int y;
    private final Font font;
    private final Color color;
    private String text;

    public TextOverlay(int x, int y, Font font, Color color, String text) {
        this.x = x;
        this.y = y;
        this.font = font;
        this.color = color;
        this.text = text;
    }

    /** Replaces the text. */
    public void setText(String text) {
        this.text = text;
    }

    /** Draws the text with its top-left corner at the configured position. */
    public void draw(Graphics2D g) {
        if (text == null || text.isEmpty()) {
            return;
        }
        g.setFont(font);
        g.setColor(color);
        g.drawString(text, x, y + g.getFontMetrics().getAscent());
    }
}
