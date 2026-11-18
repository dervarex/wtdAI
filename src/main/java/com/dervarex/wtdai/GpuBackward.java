package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Backward steps on GPU
 */
public class GpuBackward {

    static final int ROWS = 256, HIDDEN = 64, VOCAB = 2000;

    /**
     * dh = dlogits * w2 transposed, so dh[i][t] = sum over j of dlogits[i][j] * w2[t][j]
     * @param dlogits gradient of the logits, ROWS * VOCAB
     * @param w2 output weights, HIDDEN * VOCAB
     * @param dh gradient of the hidden layer, ROWS * HIDDEN
     * @param unused not used, but do not remove!!
     */
    public static void hiddenGrad(float[] dlogits, float[] w2, float[] dh, int unused) {
        for (@Parallel int i = 0; i < ROWS; i++) {
            for (@Parallel int t = 0; t < HIDDEN; t++) {
                float sum = 0f;
                for (int j = 0; j < VOCAB; j++) sum += dlogits[i * VOCAB + j] * w2[t * VOCAB + j];
                dh[i * HIDDEN + t] = sum;
            }
        }
    }

    /**
     * @return the largest difference
     */
    public static double check() {
        Random rng = new Random(8);
        float[] dlogits = new float[ROWS * VOCAB];
        float[] w2 = new float[HIDDEN * VOCAB];
        for (int i = 0; i < dlogits.length; i++) dlogits[i] = (rng.nextFloat() - 0.5f) * 0.01f;
        for (int i = 0; i < w2.length; i++) w2[i] = (rng.nextFloat() - 0.5f) * 0.2f;

        float[] dh = new float[ROWS * HIDDEN];
        TaskGraph tg = new TaskGraph("backward")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w2)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, dlogits)
                .task("hiddenGrad", GpuBackward::hiddenGrad, dlogits, w2, dh, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, dh);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        double maxDiff = 0;
        for (int i = 0; i < ROWS; i++) {
            for (int t = 0; t < HIDDEN; t++) {
                double sum = 0;
                for (int j = 0; j < VOCAB; j++) sum += dlogits[i * VOCAB + j] * w2[t * VOCAB + j];
                maxDiff = Math.max(maxDiff, Math.abs(dh[i * HIDDEN + t] - sum));
            }
        }
        return maxDiff;
    }
}
