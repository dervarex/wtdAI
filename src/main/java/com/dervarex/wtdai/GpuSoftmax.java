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

    static final int ROWS = GpuShape.ROWS, VOCAB = GpuShape.VOCAB, CELLS = ROWS * VOCAB;

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
     * ex[k] = exp(x[k] - max of its row)
     * @param x logits, ROWS * VOCAB
     * @param max one maximum per row
     * @param ex exp of every shifted logit, ROWS * VOCAB
     * @param unused not used, but don't remove it pls, it's needed
     */
    public static void expCells(float[] x, float[] max, float[] ex, int unused) {
        for (@Parallel int k = 0; k < CELLS; k++) {
            ex[k] = (float) Math.exp(x[k] - max[k / VOCAB]);
        }
    }

    /**
     * sums[i] = sum of the exp values in row i
     * @param ex exp of every shifted logit, ROWS * VOCAB
     * @param sums one sum per row
     * @param unused1 not used, but needed
     * @param unused2 not used, but needed
     */
    public static void rowSum(float[] ex, float[] sums, int unused1, int unused2) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            float s = 0f;
            for (int j = 0; j < VOCAB; j++) s += ex[i * VOCAB + j];
            sums[i] = s;
        }
    }

    /**
     * probs[k] = ex[k] / sum of its row
     * @param ex exp of every shifted logit, ROWS * VOCAB
     * @param sums one sum per row
     * @param probs probabilities, ROWS x VOCAB
     * @param unused not used, but don't remove pls
     */
    public static void divide(float[] ex, float[] sums, float[] probs, int unused) {
        for (@Parallel int k = 0; k < CELLS; k++) {
            probs[k] = ex[k] / sums[k / VOCAB];
        }
    }

    /**
     * Runs the softmax on GPU
     * @param x logits, ROWS * VOCAB
     * @return probabilities, ROWS * VOCAB
     */
    public static float[] softmaxOnGpu(float[] x) {
        float[] max = new float[ROWS];
        float[] ex = new float[CELLS];
        float[] sums = new float[ROWS];
        float[] probs = new float[CELLS];
        TaskGraph tg = new TaskGraph("softmax")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, max, ex, sums)
                .task("rowMax", GpuSoftmax::rowMax, x, max, 0, 0)
                .task("expCells", GpuSoftmax::expCells, x, max, ex, 0)
                .task("rowSum", GpuSoftmax::rowSum, ex, sums, 0, 0)
                .task("divide", GpuSoftmax::divide, ex, sums, probs, 0)
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
