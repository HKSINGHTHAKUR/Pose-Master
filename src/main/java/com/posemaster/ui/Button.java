package com.posemaster.ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;

/**
 * A clickable button drawn the way the original pygame game drew them: a filled rectangle with
 * an optional outline and centred text.
 */
public final class Button {

    private final Rectangle bounds;
    private final Color fill;
    private final Color textColor;
    private final Font font;
    private final String text;
    private final Color outline;

    public Button(int x, int y, int width, int height, Color fill, Color textColor,
                  Color outline, Font font, String text) {
        this.bounds = new Rectangle(x, y, width, height);
        this.fill = fill;
        this.textColor = textColor;
        this.outline = outline;
        this.font = font;
        this.text = text;
    }

    /** Returns the button's bounds. */
    public Rectangle bounds() {
        return bounds;
    }

    /** Returns whether the given point is inside the button. */
    public boolean contains(int x, int y) {
        return bounds.contains(x, y);
    }

    /** Draws the button. */
    public void draw(Graphics2D g) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        if (outline != null) {
            g.setColor(outline);
            g.fillRect(bounds.x - 2, bounds.y - 2, bounds.width + 4, bounds.height + 4);
        }
        g.setColor(fill);
        g.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);

        if (text != null && !text.isEmpty()) {
            g.setFont(font);
            g.setColor(textColor);
            int textWidth = g.getFontMetrics().stringWidth(text);
            int textHeight = g.getFontMetrics().getHeight();
            g.drawString(text, bounds.x + (bounds.width - textWidth) / 2,
                    bounds.y + (bounds.height + textHeight) / 2 - 4);
        }
    }
}
