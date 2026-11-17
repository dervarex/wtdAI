package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * What variable is what:
 *
 * <h2>Constants</h2>
 * <ul>
 *     <li>{@link #CONTEXT} - how many words context the model gets</li>
 *     <li>{@link #EMB} - size of each word embedding</li>
 *     <li>{@link #HIDDEN} - how many hidden neurons</li>
 *     <li>{@link #EPOCHS} - how often it goes over the text for training</li>
 *     <li>{@link #LR} - learning rate</li>
 *     <li>{@link #SEED} - the words to start from</li>
 * </ul>
 * <h2>Learned Parameters (basically the model)</h2>
 * <ul>
 *     <li>{@code x} - the 3 context embeddings</li>
 *     <li>{@code h} - hidden activation after tanh (mathematical formula)</li>
 *     <li>{@code s} - a single pre activation / logit (temp)</li>
 *     <li>{@code max} - largest logic</li>
 *     <li>{@code p} - logits, then probabilities, then gradient (wtf)</li>
 *     <li>{@code sum} - softmax normalizer</li>
 *     <li>{@code loss} - cross-entropy for the step</li>
 * </ul>
 * <h2>Inside step() - backward pass</h2>
 * <ul>
 *   <li>{@code p}  - after {@code p[target] -= 1}: how wrong each output score is</li>
 *   <li>{@code dh} - first: how much each hidden neuron contributed to the error;
 *                   then: same stuff, but measured before the tanh function</li>
 *   <li>{@code dx} - how much each input value contributed to the error</li>
 * </ul>
 *
 * <h2>Loop indices</h2>
 * <ul>
 *   <li>{@code c} - which context word (0..CONTEXT-1)</li>
 *   <li>{@code e} - which embedding dimension (0..EMB-1)</li>
 *   <li>{@code k} - flat input index (= c*EMB + e)</li>
 *   <li>{@code j} - which hidden neuron (0..HIDDEN-1)</li>
 *   <li>{@code v} - which vocabulary word (0..vocab-1)</li>
 *   <li>{@code t} - position in the text</li>
 * </ul>
 *
 * <b>Logit:</b> Raw score the model gives each turning before it makes percentages
 * <b>Softmax:</b> Turns raw scores into percentages that add up to 100%
 * <b>cross-entropy:</b> Measures how wrong these percentages are, gets measured by how little probability the model gave to the right word
 * */
public class LLM {

    static int CONTEXT = 3; // previous words the model's gonna look at
    static int EMB = 16; // size of each word embedding
    static int HIDDEN = 64;// hidden layer size
    static int EPOCHS = 2;
    static double LR = 0.05;
    static int TEST_EVERY = 10; // every ..th article is held back for measuring perplexity (overfitting or not?)
    static int THREADS = 1; // training threads
    static int BATCH = 1; // words per update
    static boolean GPU = false; // compute the batch stuff on the GPU (you're gonna need tornadovm launcher for it)

    static float[] emb; // vocab x EMB one row per word
    static float[] w1; // (CONTEXT*EMB) x HIDDEN one row per input
    static float[] b1; // HIDDEN
    static float[] w2; // HIDDEN x vocab one row per hidden neuron
    static float[] b2; // vocab

    static String SEED = "java is";
    static int SENTENCES = 3;
    static String SAVE_PATH = null;
    static String LOAD_PATH = null;


    /**
     * Entrypoint, reads the text, tokenizes, builds the vocabulary and starts training.
     * @param debug debug mode? (prints info)
     * @throws IOException if reading the training data fails
     */
    public static void start(boolean debug) throws IOException {
        Map<String, Integer> index;

        if (LOAD_PATH != null) {
            index = ModelIO.load(Paths.get(LOAD_PATH));
            if (debug) System.out.println("loaded " + LOAD_PATH + ", vocab: " + index.size());
        } else {
            List<String> trainTokens = new ArrayList<>();
            List<String> testTokens = new ArrayList<>();
            List<Path> files = Tokenizer.files();
            for (int i = 0; i < files.size(); i++) {
                List<String> tokens = Tokenizer.tokenize(Files.readString(files.get(i)));
                (i % TEST_EVERY == TEST_EVERY - 1 ? testTokens : trainTokens).addAll(tokens);
            }

            index = Vocabulary.buildVoc(trainTokens);
            int[] trainIds = Vocabulary.encode(trainTokens, index);
            int[] testIds = Vocabulary.encode(testTokens, index);

            if(debug) {
                System.out.println("train tokens: " + trainTokens.size() + ", test tokens: " + testTokens.size());
                System.out.println("vocab:  " + index.size());
            }

            train(trainIds, testIds, index.size(), debug);

            if (SAVE_PATH != null) {
                ModelIO.save(Paths.get(SAVE_PATH), index);
                if (debug) System.out.println("saved " + SAVE_PATH);
            }
        }

        for (int i = 0; i < SENTENCES; i++) {
            System.out.println(Generator.generate(SEED, CONTEXT, index));
        }
    }

    /**
     * Inits all weights randomly and runs the training
     * @param ids the training text encoded as an array of word IDs
     * @param testIds the held back text
     * @param vocab the total number of unique words (size of the output layer)
     * @param debug debug mode? prints epoch average loss and test perplexity
     */
    static void train(int[] ids, int[] testIds, int vocab, boolean debug) {
        Random rng = new Random(42);
        emb = randomArray(rng, vocab * EMB);
        w1 = randomArray(rng, CONTEXT * EMB * HIDDEN);
        b1 = new float[HIDDEN];
        w2 = randomArray(rng, HIDDEN * vocab);
        b2 = new float[vocab];

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // the learning rate shrinks linearly, down to 10% by the last epoch
            double lr = LR * (1 - 0.9 * epoch / EPOCHS);
            double total = trainEpoch(ids, ids.length, lr);
            if (debug) {
                System.out.printf("epoch %d, train loss %.4f, test perplexity %.2f%n",
                        epoch + 1, total / ids.length, perplexity(testIds, 0));
            }
        }
    }

    /**
     * Runs one pass over the training text, split across {@link #THREADS} threads
     * @param ids whole text as word IDs
     * @param count how many words to train on
     * @param lr learning rate for this pass
     * @return the summed loss over the pass
     */
    static double trainEpoch(int[] ids, int count, double lr) {
        if (BATCH > 1) return trainEpochBatched(ids, count, lr);
        double[] losses = new double[THREADS];
        Thread[] workers = new Thread[THREADS];
        for (int k = 0; k < THREADS; k++) {
            final int id = k;
            final int from = (int) ((long) count * id / THREADS);
            final int to = (int) ((long) count * (id + 1) / THREADS);
            workers[k] = new Thread(() -> {
                double sum = 0;
                for (int t = from; t < to; t++) sum += step(contextAt(ids, t), ids[t], (float) lr);
                losses[id] = sum;
            });
            workers[k].start();
        }
        for (Thread w : workers) {
            try {
                w.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        double total = 0;
        for (double l : losses) total += l;
        return total;
    }

    /**
     * Runs one pass over the training text in mini-batches, one weight update per batch
     * @param ids whole text as word IDs
     * @param count how many words to train on
     * @param lrD learning rate for this pass
     * @return the summed loss over the pass
     */
    static double trainEpochBatched(int[] ids, int count, double lrD) {
        float lr = (float) lrD;
        int vocab = b2.length;
        int inDim = CONTEXT * EMB;
        double total = 0;
        double[] expBuf = new double[vocab];
        for (int start = 0; start < count; start += BATCH) {
            int n = Math.min(BATCH, count - start);

            // inputs: context embeddings, one row per word
            float[][] x = new float[n][inDim];
            int[][] ctx = new int[n][];
            for (int b = 0; b < n; b++) {
                ctx[b] = contextAt(ids, start + b);
                for (int c = 0; c < CONTEXT; c++) {
                    for (int e = 0; e < EMB; e++) x[b][c * EMB + e] = emb[ctx[b][c] * EMB + e];
                }
            }

            // forward pass
            float[][] h = new float[n][HIDDEN];
            for (int b = 0; b < n; b++) {
                System.arraycopy(b1, 0, h[b], 0, HIDDEN);
                for (int k = 0; k < inDim; k++) {
                    float xk = x[b][k];
                    int row = k * HIDDEN;
                    for (int j = 0; j < HIDDEN; j++) h[b][j] += xk * w1[row + j];
                }
                for (int j = 0; j < HIDDEN; j++) h[b][j] = (float) Math.tanh(h[b][j]);
            }
            float[][] p = new float[n][vocab];
            float[] gpuLogits = null;
            if (GPU) {
                float[] hFlat = new float[n * HIDDEN];
                for (int b = 0; b < n; b++) System.arraycopy(h[b], 0, hFlat, b * HIDDEN, HIDDEN);
                gpuLogits = GpuMultiply.multiplyFlat(hFlat, n, HIDDEN, w2, vocab);
            }
            for (int b = 0; b < n; b++) {
                if (gpuLogits != null) {
                    for (int v = 0; v < vocab; v++) p[b][v] = b2[v] + gpuLogits[b * vocab + v];
                } else {
                    System.arraycopy(b2, 0, p[b], 0, vocab);
                    for (int j = 0; j < HIDDEN; j++) {
                        float hj = h[b][j];
                        int row = j * vocab;
                        for (int v = 0; v < vocab; v++) p[b][v] += hj * w2[row + v];
                    }
                }
                float max = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < vocab; v++) max = Math.max(max, p[b][v]);
                // exp in double
                double sum = 0;
                for (int v = 0; v < vocab; v++) {
                    expBuf[v] = Math.exp((double) (p[b][v] - max));
                    sum += expBuf[v];
                }
                for (int v = 0; v < vocab; v++) p[b][v] = (float) (expBuf[v] / sum);
                total += -Math.log(p[b][ids[start + b]] + 1e-12);
                p[b][ids[start + b]] -= 1;
                float inv = 1f / n;
                for (int v = 0; v < vocab; v++) p[b][v] *= inv;
            }

            // gradients flowing back, calculated before weight changes
            float[][] dh = new float[n][HIDDEN];
            for (int b = 0; b < n; b++) {
                for (int j = 0; j < HIDDEN; j++) {
                    float s = 0;
                    int row = j * vocab;
                    for (int v = 0; v < vocab; v++) s += w2[row + v] * p[b][v];
                    dh[b][j] = s * (1 - h[b][j] * h[b][j]);
                }
            }
            float[][] dx = new float[n][inDim];
            for (int b = 0; b < n; b++) {
                for (int k = 0; k < inDim; k++) {
                    float s = 0;
                    int row = k * HIDDEN;
                    for (int j = 0; j < HIDDEN; j++) s += w1[row + j] * dh[b][j];
                    dx[b][k] = s;
                }
            }

            // weight updates
            for (int b = 0; b < n; b++) {
                for (int j = 0; j < HIDDEN; j++) {
                    float scale = lr * h[b][j];
                    int row = j * vocab;
                    for (int v = 0; v < vocab; v++) w2[row + v] -= scale * p[b][v];
                }
                for (int v = 0; v < vocab; v++) b2[v] -= lr * p[b][v];
            }
            for (int b = 0; b < n; b++) {
                for (int k = 0; k < inDim; k++) {
                    float scale = lr * x[b][k];
                    int row = k * HIDDEN;
                    for (int j = 0; j < HIDDEN; j++) w1[row + j] -= scale * dh[b][j];
                }
            }
            for (int j = 0; j < HIDDEN; j++) {
                float g = 0;
                for (int b = 0; b < n; b++) g += dh[b][j];
                b1[j] -= lr * g;
            }
            for (int b = 0; b < n; b++) {
                for (int c = 0; c < CONTEXT; c++) {
                    for (int e = 0; e < EMB; e++) emb[ctx[b][c] * EMB + e] -= lr * dx[b][c * EMB + e];
                }
            }
        }
        return total;
    }

    /**
     * Measures how well the model predicts the held back part of the text
     * @param ids whole text as word IDs
     * @param from where the held back part starts
     * @return the perplexity, lower is better
     */
    static double perplexity(int[] ids, int from) {
        double nll = 0;
        for (int t = from; t < ids.length; t++) {
            double p = Generator.predict(contextAt(ids, t), CONTEXT)[ids[t]];
            nll -= Math.log(p + 1e-12);
        }
        return Math.exp(nll / (ids.length - from));
    }

    /**
     * Does one training step for a single word
     * @param ctx context
     * @param target the correct next word ID the model should have predicted
     * @param lr learning rate for this step
     * @return the cross-entropy loss
     */
    static double step(int[] ctx, int target, float lr) {
        int vocab = b2.length;

        float[] x = new float[CONTEXT * EMB];
        for (int c = 0; c < CONTEXT; c++) {
            for (int e = 0; e < EMB; e++) x[c * EMB + e] = emb[ctx[c] * EMB + e];
        }

        float[] h = new float[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            float s = b1[j];
            for (int k = 0; k < x.length; k++) s += x[k] * w1[k * HIDDEN + j];
            h[j] = (float) Math.tanh(s);
        }

        float[] p = new float[vocab];
        System.arraycopy(b2, 0, p, 0, vocab);
        for (int j = 0; j < HIDDEN; j++) {
            float hj = h[j];
            int row = j * vocab;
            for (int v = 0; v < vocab; v++) p[v] += hj * w2[row + v];
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < vocab; v++) max = Math.max(max, p[v]);
        float sum = 0;
        for (int v = 0; v < vocab; v++) {
            p[v] = (float) Math.exp(p[v] - max);
            sum += p[v];
        }
        for (int v = 0; v < vocab; v++) p[v] /= sum;
        double loss = -Math.log(p[target] + 1e-12);
        p[target] -= 1;

        float[] dh = new float[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            int row = j * vocab;
            for (int v = 0; v < vocab; v++) dh[j] += w2[row + v] * p[v];
        }
        for (int j = 0; j < HIDDEN; j++) {
            float scale = lr * h[j];
            int row = j * vocab;
            for (int v = 0; v < vocab; v++) w2[row + v] -= scale * p[v];
        }
        for (int v = 0; v < vocab; v++) b2[v] -= lr * p[v];

        for (int j = 0; j < HIDDEN; j++) dh[j] *= 1 - h[j] * h[j];

        float[] dx = new float[CONTEXT * EMB];
        for (int k = 0; k < x.length; k++) {
            int row = k * HIDDEN;
            for (int j = 0; j < HIDDEN; j++) {
                dx[k] += w1[row + j] * dh[j];
                w1[row + j] -= lr * x[k] * dh[j];
            }
        }
        for (int j = 0; j < HIDDEN; j++) b1[j] -= lr * dh[j];

        for (int c = 0; c < CONTEXT; c++) {
            for (int e = 0; e < EMB; e++) emb[ctx[c] * EMB + e] -= lr * dx[c * EMB + e];
        }
        return loss;
    }

    /**
     * Grabs the words that come before the current position, how much that is decides {@link #CONTEXT}
     * @param ids whole text as word IDs
     * @param t current position in text
     * @return an array of 3 word IDs representing the context
     */
    static int[] contextAt(int[] ids, int t) {
        int[] ctx = new int[CONTEXT];
        for (int c = 0; c < CONTEXT; c++) {
            int p = t - CONTEXT + c;
            ctx[c] = p >= 0 ? ids[p] : 0;
        }
        return ctx;
    }

    /**
     * Creates an array filled with small random numbers.
     * @param rng random number generator
     * @param size how many numbers
     * @return a new array with random values
     */
    static float[] randomArray(Random rng, int size) {
        float[] a = new float[size];
        for (int i = 0; i < size; i++) a[i] = (float) ((rng.nextDouble() - 0.5) * 0.2);
        return a;
    }
}
