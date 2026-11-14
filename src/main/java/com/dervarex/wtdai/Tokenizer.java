package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Tokenizer {
    public static String readText() throws IOException {
        StringBuilder text = new StringBuilder();
        try (var files = Files.list(Paths.get("training-data"))) {
            for (Path file : files.filter(p -> p.toString().endsWith(".txt")).sorted().toList()) {
                text.append(Files.readString(file)).append("\n");
            }
        }
        return text.toString();
    }

    public static List<String> tokenize(String text) {
        text = text.toLowerCase().replace(".", " . ").replaceAll("[^a-z' .]", " ");
        List<String> tokens = new ArrayList<>();
        for (String t : text.split("\\s+")) {
            if (!t.isEmpty()) tokens.add(t);
        }
        return tokens;
    }
}
