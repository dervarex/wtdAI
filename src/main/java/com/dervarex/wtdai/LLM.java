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

    static double[][] emb;// vocab x EMB
    static double[][] w1; // (CONTEXT*EMB) x HIDDEN
    static double[] b1; // HIDDEN
    static double[][] w2; // HIDDEN x vocab
    static double[] b2; // vocab

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
     * @param debug debug mode? prints epoch average loss here
     */
    static void train(int[] ids, int[] testIds, int vocab, boolean debug) {
        Random rng = new Random(42);
        emb = randomMatrix(rng, vocab, EMB);
        w1 = randomMatrix(rng, CONTEXT * EMB, HIDDEN);
        b1 = new double[HIDDEN];
        w2 = randomMatrix(rng, HIDDEN, vocab);
        b2 = new double[vocab];

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
     * Runs one pass over the training part of the text, split across {@link #THREADS} threads
     * @param ids whole text as word IDs
     * @param split where the training part ends
     * @param lr learning rate for this pass
     * @return the summed loss over the pass
     */
    static double trainEpoch(int[] ids, int split, double lr) {
        double[] losses = new double[THREADS];
        Thread[] workers = new Thread[THREADS];
        for (int k = 0; k < THREADS; k++) {
            final int id = k;
            final int from = (int) ((long) split * id / THREADS);
            final int to = (int) ((long) split * (id + 1) / THREADS);
            workers[k] = new Thread(() -> {
                double sum = 0;
                for (int t = from; t < to; t++) sum += step(contextAt(ids, t), ids[t], lr);
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
    static double step(int[] ctx, int target, double lr) {
        int vocab = b2.length;

        double[] x = new double[CONTEXT * EMB];
        for (int c = 0; c < CONTEXT; c++) {
            for (int e = 0; e < EMB; e++) x[c * EMB + e] = emb[ctx[c]][e];
        }

        double[] h = new double[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            double s = b1[j];
            for (int k = 0; k < x.length; k++) s += x[k] * w1[k][j];
            h[j] = Math.tanh(s);
        }

        double[] p = new double[vocab];
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocab; v++) {
            double s = b2[v];
            for (int j = 0; j < HIDDEN; j++) s += h[j] * w2[j][v];
            p[v] = s;
            max = Math.max(max, s);
        }
        double sum = 0;
        for (int v = 0; v < vocab; v++) {
            p[v] = Math.exp(p[v] - max);
            sum += p[v];
        }
        for (int v = 0; v < vocab; v++) p[v] /= sum;
        double loss = -Math.log(p[target] + 1e-12);
        p[target] -= 1;

        double[] dh = new double[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            for (int v = 0; v < vocab; v++) dh[j] += w2[j][v] * p[v];
        }
        for (int j = 0; j < HIDDEN; j++) {
            for (int v = 0; v < vocab; v++) w2[j][v] -= lr * h[j] * p[v];
        }
        for (int v = 0; v < vocab; v++) b2[v] -= lr * p[v];

        for (int j = 0; j < HIDDEN; j++) dh[j] *= 1 - h[j] * h[j];

        double[] dx = new double[CONTEXT * EMB];
        for (int k = 0; k < x.length; k++) {
            for (int j = 0; j < HIDDEN; j++) {
                dx[k] += w1[k][j] * dh[j];
                w1[k][j] -= lr * x[k] * dh[j];
            }
        }
        for (int j = 0; j < HIDDEN; j++) b1[j] -= lr * dh[j];

        for (int c = 0; c < CONTEXT; c++) {
            for (int e = 0; e < EMB; e++) emb[ctx[c]][e] -= lr * dx[c * EMB + e];
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
     * Creates a matrix filled with small random numbers.
     * @param rng random number generator
     * @param rows how many rows the matrix should have
     * @param cols how many columns the matrix should have
     * @return a new matrix with random values
     */
    static double[][] randomMatrix(Random rng, int rows, int cols) {
        double[][] m = new double[rows][cols];
        for (double[] row : m) {
            for (int j = 0; j < cols; j++) row[j] = (rng.nextDouble() - 0.5) * 0.2;
        }
        return m;
    }
}
