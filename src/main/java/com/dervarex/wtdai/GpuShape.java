package com.dervarex.wtdai;

/**
 * The sizes the gpu kernels work with
 */
final class GpuShape {
    static final int ROWS = 256; // words per batch
    static final int CONTEXT = LLM.CONTEXT;
    static final int EMB = LLM.EMB;
    static final int INNER = CONTEXT * EMB;
    static final int HIDDEN = LLM.HIDDEN;
    static final int VOCAB = LLM.b2 != null ? LLM.b2.length : Vocabulary.MAX_VOCAB;

    private GpuShape() {
    }
}
