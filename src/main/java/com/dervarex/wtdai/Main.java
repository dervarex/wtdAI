package com.dervarex.wtdai;


import com.dervarex.minified.utils.cli.Parser;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

public class Main {
    static void main(String[] args) {
        AtomicBoolean debug = new AtomicBoolean(false);

        Parser parser = new Parser();
        parser.register("debug", () -> debug.set(true));
        parser.register("epochs", (String v) -> LLM.EPOCHS = Integer.parseInt(v));
        parser.register("emb", (String v) -> LLM.EMB = Integer.parseInt(v));
        parser.register("hidden", (String v) -> LLM.HIDDEN = Integer.parseInt(v));
        parser.register("lr", (String v) -> LLM.LR = Double.parseDouble(v));
        parser.register("sentences", (String v) -> LLM.SENTENCES = Integer.parseInt(v));
        parser.register("seed", (String v) -> LLM.SEED = v);
        parser.register("save", (String v) -> LLM.SAVE_PATH = v);
        parser.register("load", (String v) -> LLM.LOAD_PATH = v);
        parser.register("context", (String v) -> LLM.CONTEXT = Integer.parseInt(v));
        parser.register("threads", (String v) -> LLM.THREADS = Integer.parseInt(v));
        parser.register("batch", (String v) -> LLM.BATCH = Integer.parseInt(v));
        parser.register("gpu", () -> LLM.GPU = true);
        parser.register("vocab", (String v) -> Vocabulary.MAX_VOCAB = Integer.parseInt(v));
        parser.register("gpu-check", () -> {
            System.out.println("gpu check max diff: " + GpuMultiply.selfCheck());
            System.out.println("gpu logits check max diff: " + GpuMultiply.logitsCheck());
            System.out.println("gpu weights resident (1.0 = yes): " + GpuWeights.residencyCheck());
            System.out.println("gpu batch ids wrong values (0 = okay): " + GpuGather.batchCheck());
            System.out.println("gpu hidden tanh max diff: " + GpuActivation.hiddenCheck());
            System.out.println("gpu hidden layer max diff: " + GpuHidden.check());
            System.out.println("gpu row max wrong rows (0 = ok): " + GpuSoftmax.rowMaxCheck());
            System.out.println("gpu softmax max diff: " + GpuSoftmax.softmaxCheck());
            System.out.println("gpu forward loss diff: " + GpuForward.check());
            System.exit(0);
        });
        parser.parse(args);

        try {
            LLM.start(debug.get());
        } catch (IOException e) {
            System.out.println("Failed to start the LLM. fix your code");
        }
    }
}
