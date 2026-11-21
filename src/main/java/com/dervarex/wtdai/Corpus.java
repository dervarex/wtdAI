package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * The training text as word IDs
 *
 * <h2>Constants</h2>
 * <ul>
 *     <li>{@link #MAX_WORDS} - stop reading files after this many words (whole files, so a bit more)</li>
 * </ul>
 */
public class Corpus {
    static long MAX_WORDS = Long.MAX_VALUE;

    final int[] train; // the text the model learns from
    final int[] test; // the held back text for measuring perplexity
    final Map<String, Integer> index; // word -> ID map
    final int files; // how many files were read

    private Corpus(int[] train, int[] test, Map<String, Integer> index, int files) {
        this.train = train;
        this.test = test;
        this.index = index;
        this.files = files;
    }

    /**
     * Reads the files, builds the vocabulary from the training part and encodes everything
     * @param files the training data files, see {@link Tokenizer#files()}
     * @param testEvery every ..th file is held back as test text
     * @return the encoded text and vocabulary
     * @throws IOException if reading a file fails
     */
    static Corpus read(List<Path> files, int testEvery) throws IOException {
        Map<String, Integer> seen = new HashMap<>();
        IntList counts = new IntList();
        IntList train = new IntList();
        IntList test = new IntList();

        int read = 0;
        while (read < files.size() && (long) train.size + test.size < MAX_WORDS) {
            boolean isTest = read % testEvery == testEvery - 1;
            IntList target = isTest ? test : train;
            for (String t : Tokenizer.tokenize(Files.readString(files.get(read)))) {
                Integer id = seen.get(t);
                if (id == null) {
                    id = counts.size;
                    seen.put(t, id);
                    counts.add(0);
                }
                if (!isTest) counts.data[id]++;
                target.add(id);
            }
            read++;
        }

        Map<String, Integer> trainCounts = new HashMap<>();
        for (var e : seen.entrySet()) {
            int c = counts.data[e.getValue()];
            if (c > 0) trainCounts.put(e.getKey(), c);
        }
        Map<String, Integer> index = Vocabulary.buildVoc(trainCounts);

        int unk = index.get(Vocabulary.UNK);
        int[] remap = new int[counts.size];
        for (var e : seen.entrySet()) remap[e.getValue()] = index.getOrDefault(e.getKey(), unk);

        return new Corpus(train.remapped(remap), test.remapped(remap), index, read);
    }

    /** a growing int array, a List of Integers would take 4 times the memory, we're smarter and do our own :) */
    private static final class IntList {
        int[] data = new int[1024];
        int size;

        void add(int v) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = v;
        }

        int[] remapped(int[] remap) {
            int[] out = new int[size];
            for (int i = 0; i < size; i++) out[i] = remap[data[i]];
            data = null;
            return out;
        }
    }
}
