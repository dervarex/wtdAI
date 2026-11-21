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
    static int MAX_VOCAB = 2000;
    static final String PAD = "<s>"; // fills the context before the text starts
    static final String UNK = "<unk>"; // stands for words not in the vocabulary
    static final String END = "."; // the end of a sentence

    /**
     * Keeps the {@link #MAX_VOCAB} most common words and gives each one an ID.
     * @param counts how often each word is in the training text
     * @return word -> ID, the special tokens come first
     */
    public static Map<String, Integer> buildVoc(Map<String, Integer> counts) {
        List<String> kept = new ArrayList<>(counts.keySet());
        kept.remove(END);
        // ties are sorted alphabetically so the vocabulary is the same every run
        kept.sort(Comparator.comparing((String w) -> -counts.get(w)).thenComparing(Comparator.naturalOrder()));
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
}
