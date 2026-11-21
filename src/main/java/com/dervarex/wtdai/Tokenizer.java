package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Tokenizer {
    static Path DATA_DIR = Paths.get("training-data"); // where the downloaded training data lives

    /**
     * Finds all .txt files in {@link #DATA_DIR} and its subfolders.
     * They're shuffled with a fixed seed: the test split is the same every run and
     * {@code --max-words} gets a bit of every folder instead of the first ones alphabetically.
     * @return the training data files
     * @throws IOException if there is no training data or reading the folder fails
     */
    public static List<Path> files() throws IOException {
        if (!Files.isDirectory(DATA_DIR)) {
            throw new IOException("no training data in " + DATA_DIR.toAbsolutePath() + ", run ./gradlew downloadData first");
        }
        List<Path> found;
        try (var files = Files.walk(DATA_DIR)) {
            found = new ArrayList<>(files.filter(p -> p.toString().endsWith(".txt") && Files.isRegularFile(p)).sorted().toList());
        }
        Collections.shuffle(found, new Random(42));
        if (found.isEmpty()) {
            throw new IOException("no .txt files in " + DATA_DIR.toAbsolutePath() + ", run ./gradlew downloadData first");
        }
        return found;
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
