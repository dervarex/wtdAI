package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * Context embeddings on GPU
 */
public class GpuEmbedding {

    static final int ROWS = 256, CONTEXT = 4, EMB = 16, VOCAB = 2000;

    /**
     * out row r holds the embeddings of the context ids of batch row r, one after the other
     * @param ids context ids, ROWS * CONTEXT
     * @param emb embedding table, VOCAB * EMB
     * @param out context embeddings, ROWS * CONTEXT * EMB
     * @param unused not used, but do not remove!!
     */
    public static void gatherContext(int[] ids, float[] emb, float[] out, int unused) {
        for (@Parallel int k = 0; k < ROWS * CONTEXT; k++) {
            for (int e = 0; e < EMB; e++) {
                out[k * EMB + e] = emb[ids[k] * EMB + e];
            }
        }
    }

    /**
     * Compares the GPU context gather with the host on random ids
     * @return the number of wrong values
     */
    public static int check() {
        Random rng = new Random(11);
        float[] emb = new float[VOCAB * EMB];
        for (int i = 0; i < emb.length; i++) emb[i] = rng.nextFloat() - 0.5f;
        int[] ids = new int[ROWS * CONTEXT];
        for (int i = 0; i < ids.length; i++) ids[i] = rng.nextInt(VOCAB);

        float[] out = new float[ROWS * CONTEXT * EMB];
        TaskGraph tg = new TaskGraph("embedding")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, emb)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, ids)
                .task("getContext", GpuEmbedding::gatherContext, ids, emb, out, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        new TornadoExecutionPlan(tg.snapshot()).execute();

        int wrong = 0;
        for (int k = 0; k < ROWS * CONTEXT; k++) {
            for (int e = 0; e < EMB; e++) {
                if (out[k * EMB + e] != emb[ids[k] * EMB + e]) wrong++;
            }
        }
        return wrong;
    }
}
