package com.dervarex.wtdai;

import java.util.*;

/**
 * Turns words into IDs (numbers).
 *
 * <p>Why? Neural networks can't calculate apple+apple </p>
 *
 * <h2>Special Tokens (0, 1, 2)</h2>
 * <ul>
 *   <li>{@code <s>} (PAD) - Fills empty context at the start of the text</li>
 *   <li>{@code <unk>} (UNK) - "Unknown". If the model sees a word that wasn't
 *       in the top {@link #MAX_VOCAB} list, it uses this ID instead</li>
 *   <li>{@code .} (END) - The period. Reserved so the model can learn to predict
 *       the end of a sentence.</li>
 * </ul>
 *
 * <h2>Constants</h2>
 * <ul>
 *   <li>{@link #MAX_VOCAB} - how many words the vocabulary can hold</li>
 * </ul>
 */
public class Vocabulary {
    static final int MAX_VOCAB = 2000;
    static final String PAD = "<s>"; // fills the context before the text starts
    static final String UNK = "<unk>"; // stands for words not in the vocabulary
    static final String END = "."; // the end of a sentence

    public static Map<String, Integer> buildVoc(List<String> tokens) {
        Map<String, Integer> counts = new HashMap<>();
        for (String t : tokens) counts.merge(t, 1, Integer::sum);

        counts.remove(END);
        List<String> kept = new ArrayList<>(counts.keySet());
        kept.sort((a, b) -> counts.get(b) - counts.get(a));
        kept = kept.subList(0, Math.min(kept.size(), MAX_VOCAB - 3));

        Map<String, Integer> index = new LinkedHashMap<>();
        index.put(PAD, 0);
        index.put(UNK, 1);
        index.put(END, 2);
        for (String w : kept) {
            if (!index.containsKey(w)) index.put(w, index.size());
        }
        return index;
    }

    public static int[] encode(List<String> tokens, Map<String, Integer> index) {
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = index.getOrDefault(tokens.get(i), index.get(UNK));
        }
        return ids;
    }
}
