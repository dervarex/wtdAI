package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Weight gradients on the GPU
 */
public class GpuWeightGrad {

    static final int ROWS = 256, INNER = 48, HIDDEN = 64, VOCAB = 2000;

    /**
     * gw2 = h transposed * dlogits
     * @param h hidden activations, ROWS * HIDDEN
     * @param dlogits gradient of the logits, ROWS * VOCAB
     * @param gw2 gradient of w2, HIDDEN * VOCAB
     * @param unused not used, but do not remove that!!
     */
    public static void w2Grad(float[] h, float[] dlogits, float[] gw2, int unused) {
        for (@Parallel int t = 0; t < HIDDEN; t++) {
            for (@Parallel int j = 0; j < VOCAB; j++) {
                float sum = 0f;
                for (int i = 0; i < ROWS; i++) sum += h[i * HIDDEN + t] * dlogits[i * VOCAB + j];
                gw2[t * VOCAB + j] = sum;
            }
        }
    }

    /**
     * gw1 = input transposed * dpre
     * @param x batch input, ROWS * INNER
     * @param dpre gradient before the tanh, ROWS * HIDDEN
     * @param gw1 gradient of w1, INNER * HIDDEN
     * @param unused not used, but do not remove it!!
     */
    public static void w1Grad(float[] x, float[] dpre, float[] gw1, int unused) {
        for (@Parallel int k = 0; k < INNER; k++) {
            for (@Parallel int t = 0; t < HIDDEN; t++) {
                float sum = 0f;
                for (int i = 0; i < ROWS; i++) sum += x[i * INNER + k] * dpre[i * HIDDEN + t];
                gw1[k * HIDDEN + t] = sum;
            }
        }
    }

    /**
     * @return the largest difference for both gradients
     */
    public static double check() {
        Random rng = new Random(9);
        float[] h = new float[ROWS * HIDDEN];
        float[] dlogits = new float[ROWS * VOCAB];
        float[] x = new float[ROWS * INNER];
        float[] dpre = new float[ROWS * HIDDEN];
        for (int i = 0; i < h.length; i++) h[i] = (rng.nextFloat() - 0.5f) * 2f;
        for (int i = 0; i < dlogits.length; i++) dlogits[i] = (rng.nextFloat() - 0.5f) * 0.01f;
        for (int i = 0; i < x.length; i++) x[i] = rng.nextFloat() - 0.5f;
        for (int i = 0; i < dpre.length; i++) dpre[i] = (rng.nextFloat() - 0.5f) * 0.01f;

        float[] gw2 = new float[HIDDEN * VOCAB];
        float[] gw1 = new float[INNER * HIDDEN];
        TaskGraph tg = new TaskGraph("weightGrad")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, h, dlogits, x, dpre)
                .task("w2Grad", GpuWeightGrad::w2Grad, h, dlogits, gw2, 0)
                .task("w1Grad", GpuWeightGrad::w1Grad, x, dpre, gw1, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, gw2, gw1);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        double maxDiff = 0;
        for (int t = 0; t < HIDDEN; t++) {
            for (int j = 0; j < VOCAB; j++) {
                double sum = 0;
                for (int i = 0; i < ROWS; i++) sum += h[i * HIDDEN + t] * dlogits[i * VOCAB + j];
                maxDiff = Math.max(maxDiff, Math.abs(gw2[t * VOCAB + j] - sum));
            }
        }
        for (int k = 0; k < INNER; k++) {
            for (int t = 0; t < HIDDEN; t++) {
                double sum = 0;
                for (int i = 0; i < ROWS; i++) sum += x[i * INNER + k] * dpre[i * HIDDEN + t];
                maxDiff = Math.max(maxDiff, Math.abs(gw1[k * HIDDEN + t] - sum));
            }
        }
        return maxDiff;
    }
}
