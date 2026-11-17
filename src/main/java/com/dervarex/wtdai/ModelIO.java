package com.dervarex.wtdai;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Saves and loads a trained model, so training can run on actions instead of frying my machine
 */
public class ModelIO {

    /**
     * Writes the model to a file
     * @param path where to write the model
     * @param index the vocabulary, word to ID
     * @throws IOException if the file can't be written
     */
    public static void save(Path path, Map<String, Integer> index) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
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
    }

    /**
     * Reads a model file and loads it
     * @param path the model file to read
     * @return the vocabulary, word to ID
     * @throws IOException if the file can't be read or was saved with another context size
     */
    public static Map<String, Integer> load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int context = in.readInt();
            if (context != LLM.CONTEXT) {
                throw new IOException("model was saved with context " + context + ", expected " + LLM.CONTEXT);
            }
            LLM.EMB = in.readInt();
            LLM.HIDDEN = in.readInt();

            int vocab = in.readInt();
            Map<String, Integer> index = new LinkedHashMap<>();
            for (int i = 0; i < vocab; i++) index.put(in.readUTF(), i);

            LLM.emb = readFloats(in, vocab * LLM.EMB);
            LLM.w1 = readFloats(in, context * LLM.EMB * LLM.HIDDEN);
            LLM.b1 = readFloats(in, LLM.HIDDEN);
            LLM.w2 = readFloats(in, LLM.HIDDEN * vocab);
            LLM.b2 = readFloats(in, vocab);
            return index;
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
