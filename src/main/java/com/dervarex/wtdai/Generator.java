package com.dervarex.wtdai;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.dervarex.wtdai.LLM.*;

public class Generator {
    static final int MAX_WORDS = 30;
    static final Random sampler = new Random(7);

    /**
     * Writes the sentence
     * @param seed the words to start from
     * @param index the vocabulary, word to ID
     * @return the seed followed by the generated words
     */
    static String generate(String seed, int contextLength, Map<String, Integer> index) {
        String[] words = new String[index.size()];
        for (Map.Entry<String, Integer> e : index.entrySet()) words[e.getValue()] = e.getKey();

        List<Integer> history = new ArrayList<>();
        for (String t : Tokenizer.tokenize(seed)) {
            history.add(index.getOrDefault(t, index.get(Vocabulary.UNK)));
        }

        StringBuilder out = new StringBuilder(seed);
        for (int i = 0; i < MAX_WORDS; i++) {
            int[] ctx = new int[contextLength];
            for (int c = 0; c < contextLength; c++) {
                int p = history.size() - contextLength + c;
                ctx[c] = p >= 0 ? history.get(p) : 0;
            }

            double[] probs = predict(ctx, contextLength);
            probs[0] = 0;
            probs[1] = 0;
            int next = sample(probs);

            if (next == 2) { // END
                out.append(".");
                break;
            }
            out.append(" ").append(words[next]);
            history.add(next);
        }
        return out.toString();
    }

    /**
     * Gives the probability of every word being the next one
     * @param ctx context word IDs
     * @param contextLength how many context words there are
     * @return one probability per vocabulary word
     */
    static double[] predict(int[] ctx, int contextLength) {
        int vocab = b2.length;

        float[] x = new float[contextLength * EMB];
        for (int c = 0; c < contextLength; c++) {
            for (int e = 0; e < EMB; e++) x[c * EMB + e] = emb[ctx[c] * EMB + e];
        }

        float[] h = new float[HIDDEN];
        for (int j = 0; j < HIDDEN; j++) {
            float s = b1[j];
            for (int k = 0; k < x.length; k++) s += x[k] * w1[k * HIDDEN + j];
            h[j] = (float) Math.tanh(s);
        }

        double[] p = new double[vocab];
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocab; v++) {
            double s = b2[v];
            for (int j = 0; j < HIDDEN; j++) s += h[j] * w2[j * vocab + v];
            p[v] = s;
            max = Math.max(max, s);
        }
        double sum = 0;
        for (int v = 0; v < vocab; v++) {
            p[v] = Math.exp(p[v] - max);
            sum += p[v];
        }
        for (int v = 0; v < vocab; v++) p[v] /= sum;
        return p;
    }

    /**
     * Picks an ID at random, weighted by its probability
     * @param probs probability of each ID
     * @return the chosen ID
     */
    static int sample(double[] probs) {
        double total = 0;
        for (double p : probs) total += p;
        double roll = sampler.nextDouble() * total;
        for (int v = 0; v < probs.length; v++) {
            roll -= probs[v];
            if (roll < 0) return v;
        }
        return probs.length - 1;
    }
}
