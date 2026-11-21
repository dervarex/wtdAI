package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * weight updates on GPU
 */
public class GpuUpdate {

    /**
     * w[i] = w[i] - lr * g[i]
     * @param w weights
     * @param g gradient of the weights
     * @param lr learning rate, one value
     * @param unused not used, do not remove!!
     */
    public static void step(float[] w, float[] g, float[] lr, int unused) {
        for (@Parallel int i = 0; i < w.length; i++) {
            w[i] = w[i] - lr[0] * g[i];
        }
    }

    /**
     * Runs the update for w1, w2 and the biases and compares with CPU
     * @return the largest difference
     */
    public static double check() {
        Random rng = new Random(10);
        int inner = GpuWeightGrad.INNER, hidden = GpuWeightGrad.HIDDEN, vocab = GpuWeightGrad.VOCAB;
        float[] w1 = randomArray(rng, inner * hidden, 0.2f);
        float[] b1 = randomArray(rng, hidden, 0.2f);
        float[] w2 = randomArray(rng, hidden * vocab, 0.2f);
        float[] b2 = randomArray(rng, vocab, 0.2f);
        float[] gw1 = randomArray(rng, inner * hidden, 0.01f);
        float[] gb1 = randomArray(rng, hidden, 0.01f);
        float[] gw2 = randomArray(rng, hidden * vocab, 0.01f);
        float[] gb2 = randomArray(rng, vocab, 0.01f);
        float[] lr = {0.05f};

        double[] expected = new double[w1.length + b1.length + w2.length + b2.length];
        int at = 0;
        for (int i = 0; i < w1.length; i++) expected[at++] = w1[i] - lr[0] * gw1[i];
        for (int i = 0; i < b1.length; i++) expected[at++] = b1[i] - lr[0] * gb1[i];
        for (int i = 0; i < w2.length; i++) expected[at++] = w2[i] - lr[0] * gw2[i];
        for (int i = 0; i < b2.length; i++) expected[at++] = b2[i] - lr[0] * gb2[i];

        TaskGraph tg = new TaskGraph("update")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w1, b1, w2, b2)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, gw1, gb1, gw2, gb2, lr)
                .task("stepW1", GpuUpdate::step, w1, gw1, lr, 0)
                .task("stepB1", GpuUpdate::step, b1, gb1, lr, 0)
                .task("stepW2", GpuUpdate::step, w2, gw2, lr, 0)
                .task("stepB2", GpuUpdate::step, b2, gb2, lr, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, w1, b1, w2, b2);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        double maxDiff = 0;
        at = 0;
        for (float v : w1) maxDiff = Math.max(maxDiff, Math.abs(v - expected[at++]));
        for (float v : b1) maxDiff = Math.max(maxDiff, Math.abs(v - expected[at++]));
        for (float v : w2) maxDiff = Math.max(maxDiff, Math.abs(v - expected[at++]));
        for (float v : b2) maxDiff = Math.max(maxDiff, Math.abs(v - expected[at++]));
        return maxDiff;
    }

    private static float[] randomArray(Random rng, int size, float scale) {
        float[] a = new float[size];
        for (int i = 0; i < size; i++) a[i] = (rng.nextFloat() - 0.5f) * scale;
        return a;
    }
}
