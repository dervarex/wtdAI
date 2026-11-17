package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Keeps a weight array on the GPU between runs. <br>
 * Again, that stuff only works with the TornadoVM launcher
 */
public class GpuWeights {

    private final float[] weights;
    private final float[] probe = new float[1];
    private final TornadoExecutionPlan plan;

    /**
     * Uploads the weights to the GPU
     * @param weights the host array
     */
    public GpuWeights(float[] weights) {
        this.weights = weights;
        TaskGraph tg = new TaskGraph("weights")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights)
                .task("firstValue", GpuWeights::firstValue, weights, probe)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, probe);
        this.plan = new TornadoExecutionPlan(tg.snapshot());
        plan.execute();
    }

    /**
     * Reads the first weight on the GPU
     * @param w the weights
     * @param out one value output
     */
    public static void firstValue(float[] w, float[] out) {
        out[0] = w[0];
    }

    /**
     * Runs the kernel again without uploading
     * @return the first weight as the GPU sees it
     */
    public float firstOnDevice() {
        plan.execute();
        return probe[0];
    }

    /**
     * Changes the host copy after the upload
     * @return 1.0 if the GPU copy stayed resident
     */
    public static float residencyCheck() {
        float[] w = {1f, 2f, 3f};
        GpuWeights gw = new GpuWeights(w);
        w[0] = 5f;
        return gw.firstOnDevice();
    }
}
