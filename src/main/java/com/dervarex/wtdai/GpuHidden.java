package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Hidden layer on the GPU
 */
public class GpuHidden {

    static final int ROWS = 256, INNER = 48, COLS = 64;

    private final int rows, inner, cols;
    private final float[] w1, b1, x, pre, h;
    private final TornadoExecutionPlan plan;

    /**
     * Uploads the weights and makes task graph
     * @param w1 weights, inner * cols
     * @param b1 bias, cols
     * @param rows batch size
     * @param inner input size
     * @param cols hidden size
     */
    public GpuHidden(float[] w1, float[] b1, int rows, int inner, int cols) {
        if (rows != ROWS || inner != INNER || cols != COLS) {
            throw new IllegalArgumentException("shape must be " + ROWS + " x " + INNER + " x " + COLS);
        }
        this.w1 = w1;
        this.b1 = b1;
        this.rows = rows;
        this.inner = inner;
        this.cols = cols;
        this.x = new float[rows * inner];
        this.pre = new float[rows * cols];
        this.h = new float[rows * cols];
        TaskGraph tg = new TaskGraph("hidden")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w1, b1)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, pre)
                .task("hiddenProduct", GpuHidden::hiddenProduct, x, w1, pre, 0)
                .task("biasTanh", GpuActivation::biasTanh, pre, b1, h, cols)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, h);
        this.plan = new TornadoExecutionPlan(tg.snapshot());
    }

    /**
     * c = a * b on flat row major arrays
     * @param a input, ROWS * INNER
     * @param b weights, INNER * COLS
     * @param c output, ROWS * COLS
     * @param unused not used, kept for the four argument task
     */
    public static void hiddenProduct(float[] a, float[] b, float[] c, int unused) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            for (@Parallel int j = 0; j < COLS; j++) {
                float sum = 0f;
                for (int t = 0; t < INNER; t++) sum += a[i * INNER + t] * b[t * COLS + j];
                c[i * COLS + j] = sum;
            }
        }
    }

    /**
     * Runs the hidden layer on the GPU
     * @param input batch input, rows x inner
     * @return activations, rows* cols, reused between calls
     */
    public float[] runHiddenLayer(float[] input) {
        System.arraycopy(input, 0, x, 0, x.length);
        plan.execute();
        return h;
    }

    /**
     * Compares the GPU hidden layer with CPU
     * @return the largest difference
     */
    public static double check() {
        int rows = ROWS, inner = INNER, cols = COLS;
        Random rng = new Random(5);
        float[] w1 = new float[inner * cols];
        float[] b1 = new float[cols];
        float[] input = new float[rows * inner];
        for (int i = 0; i < w1.length; i++) w1[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int j = 0; j < cols; j++) b1[j] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int i = 0; i < input.length; i++) input[i] = rng.nextFloat() - 0.5f;

        GpuHidden hidden = new GpuHidden(w1, b1, rows, inner, cols);
        float[] gpu = hidden.runHiddenLayer(input);

        double maxDiff = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = b1[j];
                for (int t = 0; t < inner; t++) sum += input[i * inner + t] * w1[t * cols + j];
                maxDiff = Math.max(maxDiff, Math.abs(gpu[i * cols + j] - Math.tanh(sum)));
            }
        }
        return maxDiff;
    }
}
