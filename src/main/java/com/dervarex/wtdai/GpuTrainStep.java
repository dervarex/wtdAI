package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;

/**
 * A training step on the gpu
 */
public class GpuTrainStep {

    static final int ROWS = 256, CONTEXT = 3, EMB = 16, INNER = CONTEXT * EMB, HIDDEN = 64, VOCAB = 2000;
    static final int CELLS = ROWS * VOCAB;

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

}
