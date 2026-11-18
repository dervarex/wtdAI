package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * A training step on the gpu
 */
public class GpuTrainStep {

    static final int ROWS = 256, CONTEXT = 3, EMB = 16, INNER = CONTEXT * EMB, HIDDEN = 64, VOCAB = 2000;
    static final int CELLS = ROWS * VOCAB;

    private final int[] ids = new int[ROWS * CONTEXT];
    private final int[] targets = new int[ROWS];
    private final float[] lr = new float[1];
    private final TornadoExecutionPlan plan;

    /**
     * x row r holds the embeddings of its context ids, one after the other
     * @param ids context ids, ROWS * CONTEXT
     * @param emb embedding table, VOCAB * EMB
     * @param x context embeddings, ROWS * INNER
     * @param unused not used, but do not remove!!
     */
    public static void getContext(int[] ids, float[] emb, float[] x, int unused) {
        for (@Parallel int r = 0; r < ROWS; r++) {
            for (int c = 0; c < CONTEXT; c++) {
                for (int e = 0; e < EMB; e++) {
                    x[r * INNER + c * EMB + e] = emb[ids[r * CONTEXT + c] * EMB + e];
                }
            }
        }
    }

    /**
     * dlogits = (probs - one hot target) / ROWS
     * @param probs probabilities, ROWS * VOCAB
     * @param targets right answer per row
     * @param dl gradient of the logits, ROWS * VOCAB
     * @param unused not used, but don't remove!
     */
    public static void gradLogits(float[] probs, int[] targets, float[] dl, int unused) {
        for (@Parallel int k = 0; k < CELLS; k++) {
            int r = k / VOCAB;
            int j = k % VOCAB;
            float onehot = j == targets[r] ? 1f : 0f;
            dl[k] = (probs[k] - onehot) / ROWS;
        }
    }

    /**
     * dpre = dh * (1 - h * h)
     * @param dh gradient of the hidden activations, ROWS * HIDDEN
     * @param h hidden activations, ROWS * HIDDEN
     * @param dpre gradient before the tanh, ROWS * HIDDEN
     * @param unused not used, but do not remove!
     */
    public static void gradHidden(float[] dh, float[] h, float[] dpre, int unused) {
        for (@Parallel int k = 0; k < ROWS * HIDDEN; k++) {
            dpre[k] = dh[k] * (1f - h[k] * h[k]);
        }
    }

    /**
     * gb1[t] = sum over the rows of dpre[row][t]
     * @param dpre gradient before the tanh, ROWS * HIDDEN
     * @param gb1 gradient of b1, HIDDEN
     * @param unused1 not used, but do not remove!
     * @param unused2 not used, but do not remove!
     */
    public static void colSumHidden(float[] dpre, float[] gb1, int unused1, int unused2) {
        for (@Parallel int t = 0; t < HIDDEN; t++) {
            float sum = 0f;
            for (int i = 0; i < ROWS; i++) sum += dpre[i * HIDDEN + t];
            gb1[t] = sum;
        }
    }

    /**
     * gb2[j] = sum over the rows of dlogits[row][j]
     * @param dl gradient of the logits, ROWS * VOCAB
     * @param gb2 gradient of b2, VOCAB
     * @param unused1 not used, but do not remove!
     * @param unused2 not used, but do not remove!
     */
    public static void colSumVocab(float[] dl, float[] gb2, int unused1, int unused2) {
        for (@Parallel int j = 0; j < VOCAB; j++) {
            float sum = 0f;
            for (int i = 0; i < ROWS; i++) sum += dl[i * VOCAB + j];
            gb2[j] = sum;
        }
    }

    /**
     * dx = dpre * w1 transposed
     * @param dpre gradient before the tanh, ROWS * HIDDEN
     * @param w1 weights, INNER * HIDDEN
     * @param dx gradient of the context embeddings, ROWS * INNER
     * @param unused not used, but don't remove!
     */
    public static void inputGrad(float[] dpre, float[] w1, float[] dx, int unused) {
        for (@Parallel int r = 0; r < ROWS; r++) {
            for (int k = 0; k < INNER; k++) {
                float sum = 0f;
                for (int t = 0; t < HIDDEN; t++) sum += dpre[r * HIDDEN + t] * w1[k * HIDDEN + t];
                dx[r * INNER + k] = sum;
            }
        }
    }

    /**
     * @param ids context ids, ROWS * CONTEXT
     * @param dx gradient of the context embeddings, ROWS * INNER
     * @param emb embedding table, VOCAB * EMB, updated in place
     * @param lr learning rate, one value
     */
    public static void embScatter(int[] ids, float[] dx, float[] emb, float[] lr) {
        for (@Parallel int v = 0; v < VOCAB; v++) {
            for (@Parallel int e = 0; e < EMB; e++) {
                float sum = 0f;
                for (int k = 0; k < ROWS * CONTEXT; k++) {
                    if (ids[k] == v) sum += dx[(k / CONTEXT) * INNER + (k % CONTEXT) * EMB + e];
                }
                emb[v * EMB + e] = emb[v * EMB + e] - lr[0] * sum;
            }
        }
    }

    /**
     * Uploads the model and makes the task graph
     * @param emb embedding table, VOCAB * EMB
     * @param w1 weights, INNER * HIDDEN
     * @param b1 bias, HIDDEN
     * @param w2 weights, HIDDEN * VOCAB
     * @param b2 bias, VOCAB
     */
    public GpuTrainStep(float[] emb, float[] w1, float[] b1, float[] w2, float[] b2) {
        if (emb.length != VOCAB * EMB || w1.length != INNER * HIDDEN || b1.length != HIDDEN
                || w2.length != HIDDEN * VOCAB || b2.length != VOCAB) {
            throw new IllegalArgumentException("gpu needs vocab " + VOCAB + ", emb " + EMB
                    + ", context " + CONTEXT + ", hidden " + HIDDEN);
        }
        float[] x = new float[ROWS * INNER];
        float[] pre = new float[ROWS * HIDDEN];
        float[] h = new float[ROWS * HIDDEN];
        float[] logits = new float[CELLS];
        float[] max = new float[ROWS];
        float[] sums = new float[ROWS];
        float[] probs = new float[CELLS];
        float[] dl = new float[CELLS];
        float[] dh = new float[ROWS * HIDDEN];
        float[] dpre = new float[ROWS * HIDDEN];
        float[] gw1 = new float[INNER * HIDDEN];
        float[] gb1 = new float[HIDDEN];
        float[] gw2 = new float[HIDDEN * VOCAB];
        float[] gb2 = new float[VOCAB];
        float[] dx = new float[ROWS * INNER];

        TaskGraph tg = new TaskGraph("trainStep")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, emb, w1, b1, w2, b2)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, ids, targets, lr)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, pre, h, logits, max, sums, probs,
                        dl, dh, dpre, gw1, gb1, gw2, gb2, dx)
                .task("gather", GpuTrainStep::getContext, ids, emb, x, 0)
                .task("hiddenProduct", GpuHidden::hiddenProduct, x, w1, pre, 0)
                .task("biasTanh", GpuActivation::biasTanh, pre, b1, h, HIDDEN)
                .task("logitsProduct", GpuForward::logitsProduct, h, w2, logits, b2)
                .task("rowMax", GpuSoftmax::rowMax, logits, max, 0, 0)
                .task("rowSumExp", GpuSoftmax::rowSumExp, logits, max, sums, 0)
                .task("divide", GpuSoftmax::divide, logits, max, sums, probs)
                .task("gradLogits", GpuTrainStep::gradLogits, probs, targets, dl, 0)
                .task("hiddenGrad", GpuBackward::hiddenGrad, dl, w2, dh, 0)
                .task("gradHidden", GpuTrainStep::gradHidden, dh, h, dpre, 0)
                .task("w2Grad", GpuWeightGrad::w2Grad, h, dl, gw2, 0)
                .task("w1Grad", GpuWeightGrad::w1Grad, x, dpre, gw1, 0)
                .task("colSumHidden", GpuTrainStep::colSumHidden, dpre, gb1, 0, 0)
                .task("colSumVocab", GpuTrainStep::colSumVocab, dl, gb2, 0, 0)
                .task("inputGrad", GpuTrainStep::inputGrad, dpre, w1, dx, 0)
                .task("embScatter", GpuTrainStep::embScatter, ids, dx, emb, lr)
                .task("stepW1", GpuUpdate::step, w1, gw1, lr, 0)
                .task("stepB1", GpuUpdate::step, b1, gb1, lr, 0)
                .task("stepW2", GpuUpdate::step, w2, gw2, lr, 0)
                .task("stepB2", GpuUpdate::step, b2, gb2, lr, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, emb, w1, b1, w2, b2);
        this.plan = new TornadoExecutionPlan(tg.snapshot());
    }

    /**
     * @param batchIds context ids, ROWS * CONTEXT
     * @param batchTargets right answer per row, ROWS
     * @param rate learning rate
     */
    public void run(int[] batchIds, int[] batchTargets, float rate) {
        System.arraycopy(batchIds, 0, ids, 0, ids.length);
        System.arraycopy(batchTargets, 0, targets, 0, targets.length);
        lr[0] = rate;
        plan.execute();
    }
}
