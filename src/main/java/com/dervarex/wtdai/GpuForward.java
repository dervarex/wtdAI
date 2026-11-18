package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Forward pass on the GPU
 */
public class GpuForward {

    static final int ROWS = 256, INNER = 48, HIDDEN = 64, VOCAB = 2000;

    private final float[] w1;
    private final float[] w2;
    private final float[] b1;
    private final float[] b2;

    private final float[] x;
    private final float[] pre;
    private final float[] h;
    private final float[] logits;
    private final float[] max;
    private final float[] sums;
    private final float[] probs;

    private final TornadoExecutionPlan plan;

    /**
     * Uploads weights and makes task graph
     * @param w1 weights, INNER * HIDDEN
     * @param b1 bias, HIDDEN
     * @param w2 weights, HIDDEN * VOCAB
     * @param b2 bias, VOCAB
     */
    public GpuForward(float[] w1, float[] b1, float[] w2, float[] b2) {
        this.w1 = w1;
        this.b1 = b1;
        this.w2 = w2;
        this.b2 = b2;
        this.x = new float[ROWS * INNER];
        this.pre = new float[ROWS * HIDDEN];
        this.h = new float[ROWS * HIDDEN];
        this.logits = new float[ROWS * VOCAB];
        this.max = new float[ROWS];
        this.sums = new float[ROWS];
        this.probs = new float[ROWS * VOCAB];
        TaskGraph tg = new TaskGraph("forward")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w1, b1, w2, b2)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, pre, h, logits, max, sums)
                .task("hiddenProduct", GpuHidden::hiddenProduct, x, w1, pre, 0)
                .task("biasTanh", GpuActivation::biasTanh, pre, b1, h, HIDDEN)
                .task("logitsProduct", GpuForward::logitsProduct, h, w2, logits, b2)
                .task("rowMax", GpuSoftmax::rowMax, logits, max, 0, 0)
                .task("rowSumExp", GpuSoftmax::rowSumExp, logits, max, sums, 0)
                .task("divide", GpuSoftmax::divide, logits, max, sums, probs)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, probs);
        this.plan = new TornadoExecutionPlan(tg.snapshot());
    }

    /**
     * logits = h * w2 + b2
     * @param h hidden activations, ROWS * HIDDEN
     * @param w2 output weights, HIDDEN * VOCAB
     * @param logits output, ROWS * VOCAB
     * @param b2 output bias, VOCAB
     */
    public static void logitsProduct(float[] h, float[] w2, float[] logits, float[] b2) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            for (@Parallel int j = 0; j < VOCAB; j++) {
                float sum = b2[j];
                for (int t = 0; t < HIDDEN; t++) sum += h[i * HIDDEN + t] * w2[t * VOCAB + j];
                logits[i * VOCAB + j] = sum;
            }
        }
    }

    /**
     * Runs forward pass
     * @param input batch input, ROWS * INNER
     * @return probabilities, ROWS * VOCAB
     */
    public float[] runForward(float[] input) {
        System.arraycopy(input, 0, x, 0, x.length);
        plan.execute();
        return probs;
    }

    /**
     * Compares GPU loss with the CPU loss using random data
     * @return the diff between GPU and CPU loss
     */
    public static double check() {
        Random rng = new Random(7);
        float[] w1 = new float[INNER * HIDDEN];
        float[] b1 = new float[HIDDEN];
        float[] w2 = new float[HIDDEN * VOCAB];
        float[] b2 = new float[VOCAB];
        float[] input = new float[ROWS * INNER];
        int[] targets = new int[ROWS];
        for (int i = 0; i < w1.length; i++) w1[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int j = 0; j < HIDDEN; j++) b1[j] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int i = 0; i < w2.length; i++) w2[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int j = 0; j < VOCAB; j++) b2[j] = (rng.nextFloat() - 0.5f) * 0.2f;
        for (int i = 0; i < input.length; i++) input[i] = rng.nextFloat() - 0.5f;
        for (int i = 0; i < ROWS; i++) targets[i] = rng.nextInt(VOCAB);

        GpuForward forward = new GpuForward(w1, b1, w2, b2);
        float[] gpuProbs = forward.runForward(input);
        double gpuLoss = loss(gpuProbs, targets);

        double[] cpuProbs = cpuForward(input, w1, b1, w2, b2);
        double cpuLoss = loss(cpuProbs, targets);

        return Math.abs(gpuLoss - cpuLoss);
    }

    /**
     * @param probs probabilities, ROWS * VOCAB
     * @param targets right answer per row
     * @return the mean loss
     */
    static double loss(float[] probs, int[] targets) {
        double total = 0;
        for (int i = 0; i < ROWS; i++) total -= Math.log(probs[i * VOCAB + targets[i]]);
        return total / ROWS;
    }

    /**
     * @param probs probabilities, ROWS * VOCAB
     * @param targets right answer per row
     * @return the mean loss
     */
    static double loss(double[] probs, int[] targets) {
        double total = 0;
        for (int i = 0; i < ROWS; i++) total -= Math.log(probs[i * VOCAB + targets[i]]);
        return total / ROWS;
    }

    /**
     * Forward pass in double on CPU
     * @return probabilities, ROWS * VOCAB
     */
    static double[] cpuForward(float[] input, float[] w1, float[] b1, float[] w2, float[] b2) {
        double[] out = new double[ROWS * VOCAB];
        for (int i = 0; i < ROWS; i++) {
            double[] hid = new double[HIDDEN];
            for (int j = 0; j < HIDDEN; j++) {
                double s = b1[j];
                for (int t = 0; t < INNER; t++) s += input[i * INNER + t] * w1[t * HIDDEN + j];
                hid[j] = Math.tanh(s);
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < VOCAB; j++) {
                double s = b2[j];
                for (int t = 0; t < HIDDEN; t++) s += hid[t] * w2[t * VOCAB + j];
                out[i * VOCAB + j] = s;
                max = Math.max(max, s);
            }
            double sum = 0;
            for (int j = 0; j < VOCAB; j++) {
                out[i * VOCAB + j] = Math.exp(out[i * VOCAB + j] - max);
                sum += out[i * VOCAB + j];
            }
            for (int j = 0; j < VOCAB; j++) out[i * VOCAB + j] /= sum;
        }
        return out;
    }
}
