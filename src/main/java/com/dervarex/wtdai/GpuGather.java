package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Reads table rows for a batch of ids on the GPU
 */
public class GpuGather {

    /**
     * out[i] = table[ids[i]]
     * @param ids batch ids
     * @param table the table, one value per id
     * @param out one value per batch entry
     * @param unused kept so the kernel fits the four argument task
     */
    public static void gather(int[] ids, float[] table, float[] out, int unused) {
        for (@Parallel int i = 0; i < ids.length; i++) {
            out[i] = table[ids[i]];
        }
    }

    /**
     * Gathers rows for the given ids on the GPU
     * @param table the table, uploaded once
     * @param ids the batch ids, uploaded on every call
     * @return one value per id
     */
    public static float[] run(float[] table, int[] ids) {
        float[] out = new float[ids.length];
        TaskGraph tg = new TaskGraph("gather")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, table)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, ids)
                .task("gather", GpuGather::gather, ids, table, out, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        new TornadoExecutionPlan(tg.snapshot()).execute();
        return out;
    }

    /**
     * Runs two batches with different ids and compares both with the host
     * @return the number of wrong values
     */
    public static int batchCheck() {
        Random rng = new Random(3);
        float[] table = new float[2000];
        for (int i = 0; i < table.length; i++) table[i] = rng.nextFloat();
        int wrong = 0;
        for (int round = 0; round < 2; round++) {
            int[] ids = new int[256];
            for (int i = 0; i < ids.length; i++) ids[i] = rng.nextInt(table.length);
            float[] out = run(table, ids);
            for (int i = 0; i < ids.length; i++) if (out[i] != table[ids[i]]) wrong++;
        }
        return wrong;
    }
}
