package com.posemaster.ml;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Classifies a pose from its MediaPipe-style landmark feature vector.
 *
 * <p>The trained Keras model shipped with the project ({@code model.h5}) was exported to ONNX by
 * {@code tools/convert_model_to_onnx.py}; {@code tools/verify_onnx.py} checks that the exported
 * graph reproduces the Keras predictions exactly.  Inference runs on ONNX Runtime, so the Java
 * application needs no Python runtime.
 */
public final class OnnxPoseClassifier implements PoseClassifier, AutoCloseable {

    /** Number of feature values the model expects (33 landmarks x,y relative to landmark 0). */
    public static final int FEATURE_COUNT = 66;

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final String inputName;
    private final String outputName;
    private final List<String> labels;

    private OnnxPoseClassifier(OrtEnvironment environment, OrtSession session, List<String> labels)
            throws OrtException {
        this.environment = environment;
        this.session = session;
        this.labels = labels;
        this.inputName = session.getInputInfo().keySet().iterator().next();
        this.outputName = session.getOutputInfo().keySet().iterator().next();
    }

    /** Loads the classifier from a model file and a labels file on disk. */
    public static OnnxPoseClassifier create(Path modelPath, Path labelsPath) throws OrtException, IOException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(1);
            OrtSession session = environment.createSession(modelPath.toString(), options);
            List<String> labels;
            try (InputStream in = Files.newInputStream(labelsPath)) {
                labels = PoseLabels.load(in);
            }
            TensorInfo output = (TensorInfo) session.getOutputInfo().values().iterator().next().getInfo();
            long[] shape = output.getShape();
            int classCount = shape.length > 1 ? (int) shape[1] : 0;
            if (classCount > 0 && classCount != labels.size()) {
                session.close();
                throw new IOException("Model outputs " + classCount + " classes but labels.npy holds "
                        + labels.size());
            }
            return new OnnxPoseClassifier(environment, session, labels);
        }
    }

    @Override
    public String classify(float[] features) {
        if (features.length != FEATURE_COUNT) {
            throw new IllegalArgumentException("Expected " + FEATURE_COUNT + " features, got " + features.length);
        }
        float[] probabilities = predict(features);
        int best = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return labels.get(best);
    }

    @Override
    public float[] predict(float[] features) {
        try (OnnxTensor input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(features),
                new long[]{1, FEATURE_COUNT});
             OrtSession.Result result = session.run(java.util.Map.of(inputName, input))) {
            float[][] output = (float[][]) result.get(0).getValue();
            return output[0];
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed", e);
        }
    }

    @Override
    public List<String> labels() {
        return labels;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException ignored) {
            // Closing a session twice is not an error worth surfacing.
        }
    }
}
