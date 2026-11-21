package com.dervarex.wtdai;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Saves and loads a trained model, so training can run on actions instead of frying my machine
 */
public class ModelIO {
    static final int FORMAT = -2; // negative, old files start with the context size, so they still load

    /**
     * A loaded model
     * @param index the vocabulary, word to ID
     * @param epochs how many epochs it was trained, 0 for old files
     */
    record Model(Map<String, Integer> index, int epochs) {
    }

    /**
     * Writes the model to a file
     * @param path where to write the model
     * @param index the vocabulary, word to ID
     * @param epochs how many epochs the model has been trained
     * @throws IOException if the file can't be written
     */
    public static void save(Path path, Map<String, Integer> index, int epochs) throws IOException {
        Path part = path.resolveSibling(path.getFileName() + ".part");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(part)))) {
            out.writeInt(FORMAT);
            out.writeInt(epochs);
            out.writeInt(LLM.CONTEXT);
            out.writeInt(LLM.EMB);
            out.writeInt(LLM.HIDDEN);

            String[] words = new String[index.size()];
            for (Map.Entry<String, Integer> e : index.entrySet()) words[e.getValue()] = e.getKey();
            out.writeInt(words.length);
            for (String w : words) out.writeUTF(w);

            writeFloats(out, LLM.emb);
            writeFloats(out, LLM.w1);
            writeFloats(out, LLM.b1);
            writeFloats(out, LLM.w2);
            writeFloats(out, LLM.b2);
        }
        Files.move(part, path, StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * Reads a model file and loads it
     * @param path the model file to read
     * @return the vocabulary and how many epochs it was trained
     * @throws IOException if the file can't be read
     */
    public static Model load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int first = in.readInt();
            int epochs = 0;
            if (first == FORMAT) {
                epochs = in.readInt();
                first = in.readInt();
            }
            LLM.CONTEXT = first;
            LLM.EMB = in.readInt();
            LLM.HIDDEN = in.readInt();

            int vocab = in.readInt();
            Map<String, Integer> index = new LinkedHashMap<>();
            for (int i = 0; i < vocab; i++) index.put(in.readUTF(), i);

            LLM.emb = readFloats(in, vocab * LLM.EMB);
            LLM.w1 = readFloats(in, LLM.CONTEXT * LLM.EMB * LLM.HIDDEN);
            LLM.b1 = readFloats(in, LLM.HIDDEN);
            LLM.w2 = readFloats(in, LLM.HIDDEN * vocab);
            LLM.b2 = readFloats(in, vocab);
            return new Model(index, epochs);
        }
    }

    private static void writeFloats(DataOutputStream out, float[] a) throws IOException {
        for (float v : a) out.writeFloat(v);
    }

    private static float[] readFloats(DataInputStream in, int length) throws IOException {
        float[] a = new float[length];
        for (int i = 0; i < length; i++) a[i] = in.readFloat();
        return a;
    }
}
