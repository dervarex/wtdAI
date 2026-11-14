package com.dervarex.wtdai;

import java.io.IOException;
import java.util.*;

/**
 * wtdAI - while(true) doAI;
 * <br><br>
 * What variable is what:
 *
 * <h2>Constants</h2>
 * <ul>
 *     <li>{@link #CONTEXT} - how many words context the model gets</li>
 *     <li>{@link #EMB} - size of each word embedding</li>
 *     <li>{@link #HIDDEN} - how many hidden neurons</li>
 *     <li>{@link #EPOCHS} - how often it goes over the text for training</li>
 *     <li>{@link #LR} - learning rate</li>
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

    static final int CONTEXT = 3; // previous words the model's gonna look at
    static final int EMB = 16; // size of each word embedding
    static final int HIDDEN = 64;// hidden layer size
    static final int EPOCHS = 2;
    static final double LR = 0.05;

    static double[][] emb;// vocab x EMB
    static double[][] w1; // (CONTEXT*EMB) x HIDDEN
    static double[] b1; // HIDDEN
    static double[][] w2; // HIDDEN x vocab
    static double[] b2; // vocab


    public static void start(boolean debug) throws IOException {
        List<String> tokens = Tokenizer.tokenize(Tokenizer.readText());
        Map<String, Integer> index = Vocabulary.buildVoc(tokens);

        int[] ids = Vocabulary.encode(tokens, index);

        if(debug) {
            System.out.println("tokens: " + tokens.size());
            System.out.println("vocab:  " + index.size());
        }

        train(ids, index.size(), debug);
    }

    static void train(int[] ids, int vocab, boolean debug) {
        Random rng = new Random(42);
        emb = randomMatrix(rng, vocab, EMB);
        w1 = randomMatrix(rng, CONTEXT * EMB, HIDDEN);
        b1 = new double[HIDDEN];
        w2 = randomMatrix(rng, HIDDEN, vocab);
        b2 = new double[vocab];

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double total = 0;
            for (int t = 0; t < ids.length; t++) {
                total += step(contextAt(ids, t), ids[t]);
            }
            if (debug) {
                System.out.printf("epoch %d, average loss %.4f%n", epoch + 1, total / ids.length);
            }
        }
    }

    static double step(int[] ctx, int target) {
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
            for (int v = 0; v < vocab; v++) w2[j][v] -= LR * h[j] * p[v];
        }
        for (int v = 0; v < vocab; v++) b2[v] -= LR * p[v];

        for (int j = 0; j < HIDDEN; j++) dh[j] *= 1 - h[j] * h[j];

        double[] dx = new double[CONTEXT * EMB];
        for (int k = 0; k < x.length; k++) {
            for (int j = 0; j < HIDDEN; j++) {
                dx[k] += w1[k][j] * dh[j];
                w1[k][j] -= LR * x[k] * dh[j];
            }
        }
        for (int j = 0; j < HIDDEN; j++) b1[j] -= LR * dh[j];

        for (int c = 0; c < CONTEXT; c++) {
            for (int e = 0; e < EMB; e++) emb[ctx[c]][e] -= LR * dx[c * EMB + e];
        }
        return loss;
    }

    static int[] contextAt(int[] ids, int t) {
        int[] ctx = new int[CONTEXT];
        for (int c = 0; c < CONTEXT; c++) {
            int p = t - CONTEXT + c;
            ctx[c] = p >= 0 ? ids[p] : 0;
        }
        return ctx;
    }

    static double[][] randomMatrix(Random rng, int rows, int cols) {
        double[][] m = new double[rows][cols];
        for (double[] row : m) {
            for (int j = 0; j < cols; j++) row[j] = (rng.nextDouble() - 0.5) * 0.2;
        }
        return m;
    }
}
