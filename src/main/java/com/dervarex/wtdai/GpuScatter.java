package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Embedding updates on GPU
 */
public class GpuScatter {

    static final int ROWS = 256, CONTEXT = 4, EMB = 16, VOCAB = 2000;

    /**
     * @param ids context ids, ROWS * CONTEXT
     * @param grads gradient of every context slot, ROWS * CONTEXT * EMB
     * @param emb embedding table, VOCAB * EMB, updated in place
     * @param lr learning rate
     */
    public static void scatterUpdate(int[] ids, float[] grads, float[] emb, float[] lr) {
        for (@Parallel int v = 0; v < VOCAB; v++) {
            for (@Parallel int e = 0; e < EMB; e++) {
                float sum = 0f;
                for (int k = 0; k < ROWS * CONTEXT; k++) {
                    if (ids[k] == v) sum += grads[k * EMB + e];
                }
                emb[v * EMB + e] = emb[v * EMB + e] - lr[0] * sum;
            }
        }
    }

    /**
     * @return the largest diff
     */
    public static double check() {
        Random rng = new Random(12);
        float[] emb = new float[VOCAB * EMB];
        for (int i = 0; i < emb.length; i++) emb[i] = rng.nextFloat() - 0.5f;
        int[] ids = new int[ROWS * CONTEXT];
        for (int i = 0; i < ids.length; i++) ids[i] = rng.nextInt(50);
        float[] grads = new float[ROWS * CONTEXT * EMB];
        for (int i = 0; i < grads.length; i++) grads[i] = (rng.nextFloat() - 0.5f) * 0.01f;
        float[] lr = {0.05f};

        double[] expected = new double[emb.length];
        for (int i = 0; i < emb.length; i++) expected[i] = emb[i];
        for (int k = 0; k < ROWS * CONTEXT; k++) {
            for (int e = 0; e < EMB; e++) {
                expected[ids[k] * EMB + e] -= lr[0] * grads[k * EMB + e];
            }
        }

        TaskGraph tg = new TaskGraph("scatter")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, emb)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, ids, grads, lr)
                .task("scatterUpdate", GpuScatter::scatterUpdate, ids, grads, emb, lr)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, emb);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        double maxDiff = 0;
        for (int i = 0; i < emb.length; i++) maxDiff = Math.max(maxDiff, Math.abs(emb[i] - expected[i]));
        return maxDiff;
    }
}
