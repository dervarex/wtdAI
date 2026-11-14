package com.dervarex.wtdai;


import com.dervarex.minified.utils.cli.Parser;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

public class Main {
    static void main(String[] args) {
        AtomicBoolean debug = new AtomicBoolean(false);

        Parser parser = new Parser();
        parser.register("debug", () -> {
            debug.set(true);
        });
        parser.register("epochs", (String v) -> LLM.EPOCHS = Integer.parseInt(v));
        parser.register("emb", (String v) -> LLM.EMB = Integer.parseInt(v));
        parser.register("hidden", (String v) -> LLM.HIDDEN = Integer.parseInt(v));
        parser.register("lr", (String v) -> LLM.LR = Double.parseDouble(v));
        parser.register("sentences", (String v) -> LLM.SENTENCES = Integer.parseInt(v));
        parser.register("seed", (String v) -> LLM.SEED = v);
        parser.register("save", (String v) -> LLM.SAVE_PATH = v);
        parser.register("load", (String v) -> LLM.LOAD_PATH = v);
        parser.parse(args);

        try {
            LLM.start(debug.get());
        } catch (IOException e) {
            System.out.println("Failed to start the LLM. fix your code");
        }
    }
}
