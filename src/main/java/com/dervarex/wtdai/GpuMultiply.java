package com.dervarex.wtdai;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.util.Random;

/**
 * Matrix multiplication on the GPU through TornadoVM (for later me: it's NOT called tornadoWM!!)
 * Only runs under the TornadoVM launcher
 */
public class GpuMultiply {

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
