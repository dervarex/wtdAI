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

    static final int ROWS = 256, VOCAB = 2000;

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
     * Runs the row maximum on the GPU
     * @param x logits, ROWS * VOCAB
     */
    public static float[] rowMaxOnGpu(float[] x) {
        float[] out = new float[ROWS];
        TaskGraph tg = new TaskGraph("softmax")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x)
                .task("rowMax", GpuSoftmax::rowMax, x, out, 0, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        new TornadoExecutionPlan(tg.snapshot()).execute();
        return out;
    }

    /**
     * Compares the GPU row maximum with the CPU
     * @return how many wrong rows we have
     */
    public static int rowMaxCheck() {
        Random rng = new Random(6);
        float[] x = new float[ROWS * VOCAB];
        for (int i = 0; i < x.length; i++) x[i] = (rng.nextFloat() - 0.5f) * 10f;

        float[] gpu = rowMaxOnGpu(x);

        int wrong = 0;
        for (int i = 0; i < ROWS; i++) {
            float cpu = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < VOCAB; j++) cpu = Math.max(cpu, x[i * VOCAB + j]);
            if (gpu[i] != cpu) wrong++;
        }
        return wrong;
    }
}
