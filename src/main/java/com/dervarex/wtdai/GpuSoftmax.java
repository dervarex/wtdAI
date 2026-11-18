package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Softmax steps on the GPU
 */
public class GpuSoftmax {

    static final int ROWS = 256, VOCAB = 2000, CELLS = ROWS * VOCAB;

    /**
     * out[i] = largest value in row i
     * @param x logits, ROWS x VOCAB
     * @param out one maximum per row
     * @param unused1 not used, but needed
     * @param unused2 not used, but needed
     */
    public static void rowMax(float[] x, float[] out, int unused1, int unused2) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            float m = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < VOCAB; j++) {
                m = Math.max(m, x[i * VOCAB + j]);
            }
            out[i] = m;
        }
    }

    /**
     * sums[i] = sum of exp(x - max) over row i
     * @param x logits, ROWS * VOCAB
     * @param max one maximum per row
     * @param sums one sum per row
     * @param unused not used, but needed
     */
    public static void rowSumExp(float[] x, float[] max, float[] sums, int unused) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            float s = 0f;
            for (int j = 0; j < VOCAB; j++) {
                s += (float) Math.exp(x[i * VOCAB + j] - max[i]);
            }
            sums[i] = s;
        }
    }

    /**
     * probs[k] = exp(x[k] - max of its row) / sum of its row
     * @param x logits, ROWS * VOCAB
     * @param max one maximum per row
     * @param sums one sum per row
     * @param probs probabilities, ROWS x VOCAB
     */
    public static void divide(float[] x, float[] max, float[] sums, float[] probs) {
        for (@Parallel int k = 0; k < CELLS; k++) {
            int row = k / VOCAB;
            probs[k] = (float) Math.exp(x[k] - max[row]) / sums[row];
        }
    }

    /**
     * Runs the softmax on GPU
     * @param x logits, ROWS * VOCAB
     * @return probabilities, ROWS * VOCAB
     */
    public static float[] softmaxOnGpu(float[] x) {
        float[] max = new float[ROWS];
        float[] sums = new float[ROWS];
        float[] probs = new float[CELLS];
        TaskGraph tg = new TaskGraph("softmax")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, max, sums)
                .task("rowMax", GpuSoftmax::rowMax, x, max, 0, 0)
                .task("rowSumExp", GpuSoftmax::rowSumExp, x, max, sums, 0)
                .task("divide", GpuSoftmax::divide, x, max, sums, probs)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, probs);
        new TornadoExecutionPlan(tg.snapshot()).execute();
        return probs;
    }

    /**
     * Compares the GPU softmax with the CPU
     * @return the largest difference between a GPU and CPU probability
     */
    public static float softmaxCheck() {
        Random rng = new Random(6);
        float[] x = new float[CELLS];
        for (int i = 0; i < x.length; i++) x[i] = (rng.nextFloat() - 0.5f) * 10f;

        float[] gpu = softmaxOnGpu(x);

        float worst = 0f;
        for (int i = 0; i < ROWS; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < VOCAB; j++) max = Math.max(max, x[i * VOCAB + j]);
            double sum = 0;
            for (int j = 0; j < VOCAB; j++) sum += Math.exp(x[i * VOCAB + j] - max);
            for (int j = 0; j < VOCAB; j++) {
                float cpu = (float) (Math.exp(x[i * VOCAB + j] - max) / sum);
                worst = Math.max(worst, Math.abs(gpu[i * VOCAB + j] - cpu));
            }
        }
        return worst;
    }

    /**
     * Compares the GPU row maximum with the CPU
     * @return how many wrong rows we have
     */
    public static int rowMaxCheck() {
        Random rng = new Random(6);
        float[] x = new float[CELLS];
        for (int i = 0; i < x.length; i++) x[i] = (rng.nextFloat() - 0.5f) * 10f;

        float[] max = new float[ROWS];
        TaskGraph tg = new TaskGraph("rowMax")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .task("rowMax", GpuSoftmax::rowMax, x, max, 0, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, max);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        int wrong = 0;
        for (int i = 0; i < ROWS; i++) {
            float cpu = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < VOCAB; j++) cpu = Math.max(cpu, x[i * VOCAB + j]);
            if (max[i] != cpu) wrong++;
        }
        return wrong;
    }
}
