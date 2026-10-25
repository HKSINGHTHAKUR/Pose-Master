package com.posemaster.pose;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Detects body landmarks with the MediaPipe BlazePose models, running on ONNX Runtime.
 *
 * <p>This is the Java replacement for the Python {@code mediapipe.solutions.pose} pipeline used
 * by {@code Game.py}.  It runs the same two models as that MediaPipe graph:
 *
 * <ol>
 *   <li>the person detector, which locates the person and returns four body keypoints;</li>
 *   <li>the pose landmark model, which predicts 39 landmarks (33 pose keypoints plus 6
 *       auxiliary points) with visibility and presence.</li>
 * </ol>
 *
 * <p>The landmark ordering, the normalised-coordinate contract and the preprocessing all match
 * the Python version, so the trained classifier keeps working unchanged.
 */
public final class MediapipePoseDetector implements PoseDetector {

    /** Input size of the person detector. */
    static final int DETECTOR_SIZE = 224;
    /** Input size of the pose landmark model. */
    static final int LANDMARK_SIZE = 256;

    private static final float SCORE_THRESHOLD = 0.5f;
    private static final float NMS_THRESHOLD = 0.3f;
    private static final int TOP_K = 5000;
    private static final float PERSON_BOX_PRE_ENLARGE_FACTOR = 1.0f;

    private final OrtEnvironment environment;
    private final OrtSession detectorSession;
    private final OrtSession landmarkSession;
    private final float[] anchors;

    private MediapipePoseDetector(OrtEnvironment environment,
                                  OrtSession detectorSession,
                                  OrtSession landmarkSession) {
        this.environment = environment;
        this.detectorSession = detectorSession;
        this.landmarkSession = landmarkSession;
        this.anchors = BlazePoseAnchors.generate();
    }

    /** Loads the person detector and pose landmark models from disk. */
    public static MediapipePoseDetector create(Path detectorModel, Path landmarkModel)
            throws OrtException, IOException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(1);
        OrtSession detector = environment.createSession(detectorModel.toString(), options);
        OrtSession landmark = environment.createSession(landmarkModel.toString(), options);
        return new MediapipePoseDetector(environment, detector, landmark);
    }

    @Override
    public PoseLandmarks detect(Mat frame) {
        PersonDetection person = detectPerson(frame);
        if (person == null) {
            return null;
        }
        return detectLandmarks(frame, person);
    }

    /** A person found by the detector: a bounding box and four body keypoints. */
    private record PersonDetection(Point[] keypoints, float score) {
    }

    private PersonDetection detectPerson(Mat frame) {
        int height = frame.rows();
        int width = frame.cols();
        float scale = Math.max(height, width);

        Mat normalized = new Mat();
        Mat rgb = new Mat();
        Imgproc.cvtColor(frame, rgb, Imgproc.COLOR_BGR2RGB);
        rgb.convertTo(normalized, org.opencv.core.CvType.CV_32FC3, 1.0 / 255.0);
        // Map [0, 1] to [-1, 1] as the MediaPipe detector expects.
        Core.subtract(normalized, new Scalar(0.5, 0.5, 0.5), normalized);
        Core.multiply(normalized, new Scalar(2.0, 2.0, 2.0), normalized);

        double ratio = Math.min((double) DETECTOR_SIZE / height, (double) DETECTOR_SIZE / width);
        int newWidth = (int) (width * ratio);
        int newHeight = (int) (height * ratio);
        Mat scaled = new Mat();
        Imgproc.resize(normalized, scaled, new Size(newWidth, newHeight));

        int padX = DETECTOR_SIZE - newWidth;
        int padY = DETECTOR_SIZE - newHeight;
        int left = padX / 2;
        int top = padY / 2;
        Mat padded = new Mat();
        Core.copyMakeBorder(scaled, padded, top, padY - top, left, padX - left,
                Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

        // Pad bias is expressed in the original frame's pixels.
        int padBiasX = (int) (left / ratio);
        int padBiasY = (int) (top / ratio);

        float[] blob = new float[3 * DETECTOR_SIZE * DETECTOR_SIZE];
        int plane = DETECTOR_SIZE * DETECTOR_SIZE;
        float[] pixels = new float[plane * 3];
        padded.get(0, 0, pixels);
        for (int i = 0; i < plane; i++) {
            int p = i * 3;
            blob[i] = pixels[p];
            blob[plane + i] = pixels[p + 1];
            blob[2 * plane + i] = pixels[p + 2];
        }

        normalized.release();
        rgb.release();
        scaled.release();
        padded.release();

        float[][][] scoreTensor;
        float[][][] deltaTensor;
        try (OnnxTensor input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(blob),
                new long[]{1, 3, DETECTOR_SIZE, DETECTOR_SIZE})) {
            Map<String, OnnxTensor> inputs = Map.of(
                    detectorSession.getInputInfo().keySet().iterator().next(), input);
            try (OrtSession.Result result = detectorSession.run(inputs)) {
                // Output 0 holds the per-anchor score [1, 2254, 1], output 1 the box and
                // keypoint deltas [1, 2254, 12].
                scoreTensor = (float[][][]) result.get(0).getValue();
                deltaTensor = (float[][][]) result.get(1).getValue();
            }
        } catch (OrtException e) {
            throw new IllegalStateException("Person detection failed", e);
        }

        int anchorCount = scoreTensor[0].length;
        float[] scores = new float[anchorCount];
        float[][] deltas = new float[anchorCount][];
        for (int i = 0; i < anchorCount; i++) {
            scores[i] = scoreTensor[0][i][0];
            deltas[i] = deltaTensor[0][i];
        }

        double[] boxX1 = new double[deltas.length];
        double[] boxY1 = new double[deltas.length];
        double[] boxX2 = new double[deltas.length];
        double[] boxY2 = new double[deltas.length];
        float[] candidateScores = new float[deltas.length];
        double[][] keypointX = new double[deltas.length][4];
        double[][] keypointY = new double[deltas.length][4];

        for (int i = 0; i < deltas.length; i++) {
            float score = sigmoid(scores[i]);
            candidateScores[i] = score;

            // The reference computes (delta / inputSize + anchor) * scale, then subtracts the
            // letterbox offset.  Keep the same order of operations, in double precision.
            double cx = (deltas[i][0] / DETECTOR_SIZE + anchors[i * 2]) * scale - padBiasX;
            double cy = (deltas[i][1] / DETECTOR_SIZE + anchors[i * 2 + 1]) * scale - padBiasY;
            double w = deltas[i][2] / DETECTOR_SIZE * scale;
            double h = deltas[i][3] / DETECTOR_SIZE * scale;
            boxX1[i] = cx - w / 2;
            boxY1[i] = cy - h / 2;
            boxX2[i] = cx + w / 2;
            boxY2[i] = cy + h / 2;

            for (int k = 0; k < 4; k++) {
                keypointX[i][k] = (deltas[i][4 + k * 2] / DETECTOR_SIZE + anchors[i * 2]) * scale - padBiasX;
                keypointY[i][k] = (deltas[i][5 + k * 2] / DETECTOR_SIZE + anchors[i * 2 + 1]) * scale - padBiasY;
            }
        }

        int[] keepIndices = nonMaxSuppression(boxX1, boxY1, boxX2, boxY2, candidateScores);
        if (keepIndices.length == 0) {
            return null;
        }

        int best = keepIndices[0];
        for (int index : keepIndices) {
            if (candidateScores[index] > candidateScores[best]) {
                best = index;
            }
        }

        // Four keypoints: hip centre, full body point, shoulder centre and upper body point.
        Point[] keypoints = new Point[4];
        for (int k = 0; k < 4; k++) {
            keypoints[k] = new Point((int) keypointX[best][k], (int) keypointY[best][k]);
        }

        return new PersonDetection(keypoints, candidateScores[best]);
    }

    /**
     * Greedy non-maximum suppression over the decoded boxes.
     *
     * @return the indices of the kept boxes, ordered by descending score
     */
    private static int[] nonMaxSuppression(double[] x1, double[] y1, double[] x2, double[] y2, float[] scores) {
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] >= SCORE_THRESHOLD) {
                candidates.add(i);
            }
        }
        candidates.sort((a, b) -> Float.compare(scores[b], scores[a]));
        if (candidates.size() > TOP_K) {
            candidates = candidates.subList(0, TOP_K);
        }

        List<Integer> kept = new ArrayList<>();
        for (int index : candidates) {
            boolean suppressed = false;
            for (int other : kept) {
                if (iou(index, other, x1, y1, x2, y2) > NMS_THRESHOLD) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.add(index);
            }
        }

        int[] result = new int[kept.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = kept.get(i);
        }
        return result;
    }

    /** Intersection over union of two axis-aligned boxes. */
    private static float iou(int a, int b, double[] x1, double[] y1, double[] x2, double[] y2) {
        double left = Math.max(x1[a], x1[b]);
        double top = Math.max(y1[a], y1[b]);
        double right = Math.min(x2[a], x2[b]);
        double bottom = Math.min(y2[a], y2[b]);
        double intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
        if (intersection <= 0) {
            return 0f;
        }
        double areaA = (x2[a] - x1[a]) * (y2[a] - y1[a]);
        double areaB = (x2[b] - x1[b]) * (y2[b] - y1[b]);
        double union = areaA + areaB - intersection;
        return union <= 0 ? 0f : (float) (intersection / union);
    }

    private PoseLandmarks detectLandmarks(Mat frame, PersonDetection person) {
        Point midHip = person.keypoints()[0];
        Point fullBody = person.keypoints()[1];

        double fullDistance = Math.hypot(midHip.x - fullBody.x, midHip.y - fullBody.y);
        int[] fullBox = {
                (int) (midHip.x - fullDistance),
                (int) (midHip.y - fullDistance),
                (int) (midHip.x + fullDistance),
                (int) (midHip.y + fullDistance)
        };
        double centerX = (fullBox[0] + fullBox[2]) / 2.0;
        double centerY = (fullBox[1] + fullBox[3]) / 2.0;
        double halfWidth = (fullBox[2] - fullBox[0]) * PERSON_BOX_PRE_ENLARGE_FACTOR / 2.0;
        double halfHeight = (fullBox[3] - fullBox[1]) * PERSON_BOX_PRE_ENLARGE_FACTOR / 2.0;
        fullBox[0] = (int) (centerX - halfWidth);
        fullBox[1] = (int) (centerY - halfHeight);
        fullBox[2] = (int) (centerX + halfWidth);
        fullBox[3] = (int) (centerY + halfHeight);

        int[] clipped = {
                clamp(fullBox[0], 0, frame.cols()),
                clamp(fullBox[1], 0, frame.rows()),
                clamp(fullBox[2], 0, frame.cols()),
                clamp(fullBox[3], 0, frame.rows())
        };

        int cropX = clipped[0];
        int cropY = clipped[1];
        int cropWidth = Math.max(1, clipped[2] - clipped[0]);
        int cropHeight = Math.max(1, clipped[3] - clipped[1]);
        Mat crop = frame.submat(new Rect(cropX, cropY, cropWidth, cropHeight)).clone();

        int left = clipped[0] - fullBox[0];
        int top = clipped[1] - fullBox[1];
        int right = fullBox[2] - clipped[2];
        int bottom = fullBox[3] - clipped[3];
        Mat padded = new Mat();
        Core.copyMakeBorder(crop, padded, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
        crop.release();

        int padBiasX = clipped[0] - left;
        int padBiasY = clipped[1] - top;

        double midHipX = midHip.x - padBiasX;
        double midHipY = midHip.y - padBiasY;
        double fullBodyX = fullBody.x - padBiasX;
        double fullBodyY = fullBody.y - padBiasY;

        double radians = Math.PI / 2 - Math.atan2(-(fullBodyY - midHipY), fullBodyX - midHipX);
        radians = radians - 2 * Math.PI * Math.floor((radians + Math.PI) / (2 * Math.PI));
        double angle = Math.toDegrees(radians);

        Mat rotation = Imgproc.getRotationMatrix2D(new Point(midHipX, midHipY), angle, 1.0);
        Mat rotated = new Mat();
        Imgproc.warpAffine(padded, rotated, rotation, padded.size());
        int paddedWidth = padded.cols();
        int paddedHeight = padded.rows();
        padded.release();

        Mat resized = new Mat();
        Imgproc.resize(rotated, resized, new Size(LANDMARK_SIZE, LANDMARK_SIZE), 0, 0, Imgproc.INTER_AREA);
        rotated.release();

        float[] blob = new float[LANDMARK_SIZE * LANDMARK_SIZE * 3];
        byte[] pixels = new byte[LANDMARK_SIZE * LANDMARK_SIZE * 3];
        resized.get(0, 0, pixels);
        for (int i = 0; i < LANDMARK_SIZE * LANDMARK_SIZE; i++) {
            int p = i * 3;
            // The model expects RGB in [0, 1].
            blob[i * 3] = (pixels[p + 2] & 0xFF) / 255.0f;
            blob[i * 3 + 1] = (pixels[p + 1] & 0xFF) / 255.0f;
            blob[i * 3 + 2] = (pixels[p] & 0xFF) / 255.0f;
        }
        resized.release();

        float[][] landmarkOutput;
        float[][] confidenceOutput;
        try (OnnxTensor input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(blob),
                new long[]{1, LANDMARK_SIZE, LANDMARK_SIZE, 3})) {
            Map<String, OnnxTensor> inputs = Map.of(
                    landmarkSession.getInputInfo().keySet().iterator().next(), input);
            try (OrtSession.Result result = landmarkSession.run(inputs)) {
                // Output 0 holds the 39 landmarks flattened as [1, 195], output 1 the
                // confidence [1, 1].
                landmarkOutput = (float[][]) result.get(0).getValue();
                confidenceOutput = (float[][]) result.get(1).getValue();
            }
        } catch (OrtException e) {
            throw new IllegalStateException("Pose landmark inference failed", e);
        }

        float confidence = confidenceOutput[0][0];
        if (confidence < SCORE_THRESHOLD) {
            return null;
        }

        // 39 rows of [x, y, z, visibility, presence].
        float[] flat = landmarkOutput[0];
        float[][] raw = new float[39][5];
        for (int i = 0; i < 39 && i * 5 + 4 < flat.length; i++) {
            System.arraycopy(flat, i * 5, raw[i], 0, 5);
        }

        double scaleX = (double) paddedWidth / LANDMARK_SIZE;
        double scaleY = (double) paddedHeight / LANDMARK_SIZE;

        // The reference implementation maps the ROI pixel coordinates back to the original frame
        // with an explicit rotation matrix and its inverse.  Reproduce that algebra exactly.
        double cos = Math.cos(Math.toRadians(angle));
        double sin = Math.sin(Math.toRadians(angle));
        double m00 = cos;
        double m01 = sin;
        double m10 = -sin;
        double m11 = cos;
        double tx = (1 - cos) * midHipX - sin * midHipY;
        double ty = sin * midHipX + (1 - cos) * midHipY;

        double invTx = -(m00 * tx + m10 * ty);
        double invTy = -(m01 * tx + m11 * ty);

        double roiCenterX = paddedWidth / 2.0;
        double roiCenterY = paddedHeight / 2.0;
        double originalCenterX = roiCenterX * m00 + roiCenterY * m10 + invTx;
        double originalCenterY = roiCenterX * m01 + roiCenterY * m11 + invTy;

        List<PoseLandmark> landmarks = new ArrayList<>(PoseLandmarks.LANDMARK_COUNT);
        for (int i = 0; i < PoseLandmarks.LANDMARK_COUNT; i++) {
            float[] point = raw[i];
            double x = (point[0] - LANDMARK_SIZE / 2.0) * scaleX;
            double y = (point[1] - LANDMARK_SIZE / 2.0) * scaleY;

            // Rotate by +angle, matching the reference's coords_rotation_matrix.
            double rotatedX = x * cos - y * sin;
            double rotatedY = x * sin + y * cos;

            double frameX = rotatedX + originalCenterX + padBiasX;
            double frameY = rotatedY + originalCenterY + padBiasY;

            float visibility = sigmoid(point[3]);
            float presence = sigmoid(point[4]);

            // The Python version consumed MediaPipe's normalised coordinates, where x is divided
            // by the frame width and y by the frame height.  Keep that contract so the trained
            // classifier sees the same kind of feature vector it was fitted on.
            landmarks.add(new PoseLandmark((float) (frameX / frame.cols()),
                    (float) (frameY / frame.rows()),
                    visibility * presence));
        }

        return new PoseLandmarks(landmarks);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    @Override
    public void close() {
        try {
            detectorSession.close();
            landmarkSession.close();
        } catch (OrtException ignored) {
            // Nothing useful to do if closing fails.
        }
    }
}
