package com.posemaster.ml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads the label list that accompanies the trained pose-classification model.
 *
 * <p>The labels are stored in a NumPy {@code labels.npy} file holding a one-dimensional array of
 * fixed-width UTF-32 strings.  The file is shared with the Python version, so it is parsed
 * directly here instead of requiring NumPy.
 *
 * <p>Supported layouts: {@code <U<n>} (fixed-width unicode) and {@code <S<n>} (fixed-width
 * bytes), both little-endian and one-dimensional.  The file must be a version 1.0 or 2.0
 * {@code .npy} file.
 */
public final class PoseLabels {

    private PoseLabels() {
    }

    /** Loads the labels from a stream positioned at the start of a {@code .npy} file. */
    public static List<String> load(InputStream in) throws IOException {
        byte[] magic = readFully(in, 6);
        if (magic[0] != (byte) 0x93 || !"NUMPY".equals(new String(magic, 1, 5, StandardCharsets.US_ASCII))) {
            throw new IOException("Not a .npy file");
        }

        int major = readByte(in);
        int minor = readByte(in);
        int headerLength = (major == 1) ? readShort(in) : readInt(in);
        if (headerLength < 0 || headerLength > 1 << 20) {
            throw new IOException("Implausible .npy header length: " + headerLength);
        }

        String header = new String(readFully(in, headerLength), StandardCharsets.ISO_8859_1).trim();
        Map<String, String> meta = parseHeader(header);

        String descr = meta.get("descr");
        if (descr == null) {
            throw new IOException("Missing 'descr' in .npy header");
        }
        // The dtype carries a byte-order character: '<U15', '>U15' or '|U15'.
        String type = descr;
        if (!type.isEmpty() && "<>=|".indexOf(type.charAt(0)) >= 0) {
            type = type.substring(1);
        }
        char kind = type.charAt(0);
        int itemSize;
        if (kind == 'U' || kind == 'S') {
            itemSize = Integer.parseInt(type.substring(1));
        } else {
            throw new IOException("Unsupported .npy dtype for labels: " + descr);
        }

        int count = 1;
        String shape = meta.get("shape");
        if (shape != null) {
            String trimmed = shape.trim();
            if (trimmed.startsWith("(") && trimmed.endsWith(")")) {
                trimmed = trimmed.substring(1, trimmed.length() - 1).trim();
                if (!trimmed.isEmpty()) {
                    count = Integer.parseInt(trimmed.split(",")[0].trim());
                }
            }
        }

        int bytesPerChar = kind == 'U' ? 4 : 1;
        byte[] payload = readFully(in, count * itemSize * bytesPerChar);

        List<String> labels = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int offset = i * itemSize * bytesPerChar;
            String value;
            if (kind == 'U') {
                ByteBuffer buffer = ByteBuffer.wrap(payload, offset, itemSize * 4)
                        .order(ByteOrder.LITTLE_ENDIAN);
                StringBuilder builder = new StringBuilder(itemSize);
                for (int c = 0; c < itemSize; c++) {
                    builder.append((char) buffer.getInt());
                }
                value = builder.toString();
            } else {
                value = new String(payload, offset, itemSize, StandardCharsets.UTF_8);
            }
            // Fixed-width strings are padded with null characters.
            labels.add(stripPadding(value));
        }
        return labels;
    }

    /**
     * Parses the small Python-dict header of a {@code .npy} file.  The header uses single
     * quotes around keys, but values may be quoted strings ({@code 'descr': '<U15'}) or bare
     * Python literals ({@code 'shape': (16,)}), so it is parsed by hand.
     */
    private static Map<String, String> parseHeader(String header) {
        Map<String, String> meta = new LinkedHashMap<>();
        int i = 0;
        while (i < header.length()) {
            int keyStart = header.indexOf('\'', i);
            if (keyStart < 0) {
                break;
            }
            int keyEnd = header.indexOf('\'', keyStart + 1);
            if (keyEnd < 0) {
                break;
            }
            String key = header.substring(keyStart + 1, keyEnd);

            int colon = header.indexOf(':', keyEnd);
            if (colon < 0) {
                break;
            }
            int valueStart = colon + 1;
            while (valueStart < header.length() && Character.isWhitespace(header.charAt(valueStart))) {
                valueStart++;
            }
            if (valueStart >= header.length()) {
                break;
            }

            if (header.charAt(valueStart) == '\'') {
                int valueEnd = header.indexOf('\'', valueStart + 1);
                if (valueEnd < 0) {
                    break;
                }
                meta.put(key, header.substring(valueStart + 1, valueEnd));
                i = valueEnd + 1;
            } else {
                // A bare literal such as (16,) or False: read up to the next comma that is not
                // inside brackets.
                int depth = 0;
                int j = valueStart;
                while (j < header.length()) {
                    char c = header.charAt(j);
                    if (c == '(' || c == '[' || c == '{') {
                        depth++;
                    } else if (c == ')' || c == ']' || c == '}') {
                        depth--;
                    } else if (c == ',' && depth == 0) {
                        break;
                    }
                    j++;
                }
                meta.put(key, header.substring(valueStart, j).trim());
                i = j;
            }
        }
        return meta;
    }

    /** Removes the null padding numpy adds to fixed-width strings. */
    private static String stripPadding(String value) {
        int end = value.length();
        while (end > 0 && value.charAt(end - 1) == '\0') {
            end--;
        }
        return value.substring(0, end);
    }

    private static byte[] readFully(InputStream in, int length) throws IOException {
        byte[] buffer = new byte[length];
        int read = 0;
        while (read < length) {
            int n = in.read(buffer, read, length - read);
            if (n < 0) {
                throw new IOException("Unexpected end of .npy file");
            }
            read += n;
        }
        return buffer;
    }

    private static int readByte(InputStream in) throws IOException {
        int value = in.read();
        if (value < 0) {
            throw new IOException("Unexpected end of .npy file");
        }
        return value;
    }

    private static int readShort(InputStream in) throws IOException {
        int low = readByte(in);
        int high = readByte(in);
        return (high << 8) | low;
    }

    private static int readInt(InputStream in) throws IOException {
        int b0 = readByte(in);
        int b1 = readByte(in);
        int b2 = readByte(in);
        int b3 = readByte(in);
        return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
    }
}
