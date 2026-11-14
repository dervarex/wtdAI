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
        parser.parse(args);

        try {
            LLM.start(debug.get());
        } catch (IOException e) {
            System.out.println("Failed to start the LLM. fix your code");
        }
    }
}
