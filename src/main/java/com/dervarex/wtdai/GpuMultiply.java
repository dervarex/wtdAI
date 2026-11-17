package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Matrix multiplication on the GPU through TornadoVM (for later me: it's NOT called tornadoWM!!)
 * Only runs under the TornadoVM launcher
 */
public class GpuMultiply {

    // task graph and its device buffers per matrix shape
    private static final Map<String, Object[]> CACHE = new HashMap<>();

    /**
     * Kernel: c = a * b
     * Uses one GPU thread per output cell
     * @param a left matrix (rows * inner)
     * @param b right matrix (inner * cols)
     * @param c output matrix (rows * cols)
     * @param unused not used lol, kept so the kernel fits the four-argument task (do not remove it!)
     */
    public static void cellProduct(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat c, int unused) {
        for (@Parallel int i = 0; i < c.getNumRows(); i++) {
            for (@Parallel int j = 0; j < c.getNumColumns(); j++) {
                float sum = 0f;
                for (int k = 0; k < a.getNumColumns(); k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, sum);
            }
        }
    }

    /**
     * Multiplies two matrices on the GPU and copies the result back
     * @param a left matrix
     * @param b right matrix
     * @param c output matrix, overwritten
     */
    public static void multiply(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat c) {
        TaskGraph tg = new TaskGraph("gpu")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .task("cellProduct", GpuMultiply::cellProduct, a, b, c, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
        ImmutableTaskGraph itg = tg.snapshot();
        new TornadoExecutionPlan(itg).execute();
    }

    /**
     * Multiplies flat row major arrays on the GPU
     * @param a left matrix, rows x inner
     * @param rows rows of a
     * @param inner columns of a, rows of b
     * @param b right matrix, inner x cols
     * @param cols columns of b
     * @return the product as a row-major array of rows * cols values
     */
    public static float[] multiplyFlat(float[] a, int rows, int inner, float[] b, int cols) {
        String key = rows + "x" + inner + "x" + cols;
        Object[] cached = CACHE.get(key);
        if (cached == null) {
            Matrix2DFloat ma = new Matrix2DFloat(rows, inner);
            Matrix2DFloat mb = new Matrix2DFloat(inner, cols);
            Matrix2DFloat mc = new Matrix2DFloat(rows, cols);
            TaskGraph tg = new TaskGraph("gpu")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, ma, mb)
                    .task("cellProduct", GpuMultiply::cellProduct, ma, mb, mc, 0)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, mc);
            cached = new Object[]{ma, mb, mc, new TornadoExecutionPlan(tg.snapshot())};
            CACHE.put(key, cached);
        }
        Matrix2DFloat ma = (Matrix2DFloat) cached[0];
        Matrix2DFloat mb = (Matrix2DFloat) cached[1];
        Matrix2DFloat mc = (Matrix2DFloat) cached[2];
        for (int i = 0; i < rows; i++) for (int j = 0; j < inner; j++) ma.set(i, j, a[i * inner + j]);
        for (int i = 0; i < inner; i++) for (int j = 0; j < cols; j++) mb.set(i, j, b[i * cols + j]);
        ((TornadoExecutionPlan) cached[3]).execute();
        float[] out = new float[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) out[i * cols + j] = mc.get(i, j);
        return out;
    }

    /**
     * Checks the flat multiply at the model's logits size (batch 256, hidden 64, vocab 2000)
     * @return the largest difference between the GPU and the CPU result
     */
    public static double logitsCheck() {
        int rows = 256, inner = 64, cols = 2000;
        Random rng = new Random(2);
        float[] h = new float[rows * inner];
        float[] w2 = new float[inner * cols];
        for (int i = 0; i < h.length; i++) h[i] = rng.nextFloat() - 0.5f;
        for (int i = 0; i < w2.length; i++) w2[i] = rng.nextFloat() - 0.5f;

        float[] gpu = multiplyFlat(h, rows, inner, w2, cols);

        double maxDiff = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0f;
                for (int t = 0; t < inner; t++) sum += h[i * inner + t] * w2[t * cols + j];
                maxDiff = Math.max(maxDiff, Math.abs(gpu[i * cols + j] - sum));
            }
        }
        return maxDiff;
    }

    /**
     * Checks the GPU result to a plain CPU loop on random matrices
     * @return the largest difference between the two results
     */
    public static double selfCheck() {
        int m = 256, k = 64, n = 512;
        Random rng = new Random(1);
        Matrix2DFloat a = new Matrix2DFloat(m, k);
        Matrix2DFloat b = new Matrix2DFloat(k, n);
        Matrix2DFloat c = new Matrix2DFloat(m, n);
        for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) a.set(i, j, rng.nextFloat() - 0.5f);
        for (int i = 0; i < k; i++) for (int j = 0; j < n; j++) b.set(i, j, rng.nextFloat() - 0.5f);

        multiply(a, b, c);

        double maxDiff = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                for (int t = 0; t < k; t++) sum += a.get(i, t) * b.get(t, j);
                maxDiff = Math.max(maxDiff, Math.abs(c.get(i, j) - sum));
            }
        }
        return maxDiff;
    }
}
