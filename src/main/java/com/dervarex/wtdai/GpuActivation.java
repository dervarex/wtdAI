package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

public class GpuActivation {

    /**
     * out[i] = tanh(pre[i] + bias[i % cols])
     * @param pre pre-activations, rows * cols
     * @param bias one bias per column
     * @param out activations, rows * cols
     * @param cols number of columns
     */
    public static void biasTanh(float[] pre, float[] bias, float[] out, int cols) {
        for (@Parallel int i = 0; i < pre.length; i++) {
            out[i] = (float) Math.tanh(pre[i] + bias[i % cols]);
        }
    }

    /**
     * Runs the bias and tanh on the GPU
     * @param pre pre activations, rows * cols
     * @param bias one bias per column
     * @param cols number of columns
     * @return the activations, rows * cols
     */
    public static float[] run(float[] pre, float[] bias, int cols) {
        float[] out = new float[pre.length];
        TaskGraph tg = new TaskGraph("activation")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, pre, bias)
                .task("biasTanh", GpuActivation::biasTanh, pre, bias, out, cols)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        new TornadoExecutionPlan(tg.snapshot()).execute();
        return out;
    }

    /**
     * Compares the GPU result with the CPU on hidden layer sized data
     * @return the largest difference
     */
    public static double hiddenCheck() {
        int rows = 256, cols = 64;
        Random rng = new Random(4);
        float[] pre = new float[rows * cols];
        float[] bias = new float[cols];
        for (int i = 0; i < pre.length; i++) pre[i] = (rng.nextFloat() - 0.5f) * 4f;
        for (int j = 0; j < cols; j++) bias[j] = rng.nextFloat() - 0.5f;

        float[] gpu = run(pre, bias, cols);

        double maxDiff = 0;
        for (int i = 0; i < pre.length; i++) {
            double cpu = Math.tanh(pre[i] + bias[i % cols]);
            maxDiff = Math.max(maxDiff, Math.abs(gpu[i] - cpu));
        }
        return maxDiff;
    }
}
