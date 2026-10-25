package com.posemaster.ui;

import com.posemaster.app.ResourceLocator;

import javax.imageio.ImageIO;
import java.awt.Font;
import java.awt.FontFormatException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Loads the game's images and fonts.
 *
 * <p>All the artwork and the four bundled fonts are shared with the Python version, so both
 * versions have the same visual identity.  See {@code THIRD_PARTY_NOTICES.md} for their licenses.
 */
public final class Assets {

    /** Font file names, matching the Python version's {@code Fonts} directory. */
    public static final String FONT_MARCELLUS = "Fonts/Marcellus-Regular.ttf";
    public static final String FONT_SILKSCREEN = "Fonts/Silkscreen-Regular.ttf";
    public static final String FONT_NABLA = "Fonts/Nabla.ttf";
    public static final String FONT_GOD_OF_WAR = "Fonts/god-of-war.ttf";

    private final ResourceLocator resources;
    private final Map<String, BufferedImage> images = new HashMap<>();
    private final Map<String, Font> fonts = new HashMap<>();

    public Assets(ResourceLocator resources) {
        this.resources = resources;
    }

    /** Loads an image, caching the result. */
    public BufferedImage image(String name) throws IOException {
        BufferedImage cached = images.get(name);
        if (cached != null) {
            return cached;
        }
        try (InputStream in = resources.open(name)) {
            BufferedImage image = ImageIO.read(in);
            if (image == null) {
                throw new IOException("Unsupported image format: " + name);
            }
            images.put(name, image);
            return image;
        }
    }

    /** Loads a font, caching the result. */
    public Font font(String name, int size) throws IOException {
        String key = name + "#" + size;
        Font cached = fonts.get(key);
        if (cached != null) {
            return cached;
        }
        try (InputStream in = resources.open(name)) {
            Font font = Font.createFont(Font.TRUETYPE_FONT, in).deriveFont((float) size);
            fonts.put(key, font);
            return font;
        } catch (FontFormatException e) {
            throw new IOException("Could not load font: " + name, e);
        }
    }
}
