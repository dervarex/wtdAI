package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        StringBuilder corpus = new StringBuilder();
        try (var files = Files.list(Paths.get("training-data"))) {
            for (Path file : files.filter(p -> p.toString().endsWith(".txt")).sorted().toList()) {
                corpus.append(Files.readString(file)).append("\n");
            }
        }
        String text = corpus.toString();

        text = text.replaceAll("(?m)^#--.*$", "");
        text = text.toLowerCase();
        text = text.replace(".", " <end> ");
        text = text.replace(",", "");
        text = text.replaceAll("\\[.*?\\]", "");
        text = text.replaceAll("[^a-z ]", " ");

        String[] tokens = text.split("\\s+");

        Map<String, Map<String, Integer>> pairs = new HashMap<>();
        Map<String, Map<String, Integer>> bigrams = new HashMap<>();

        for(int i = 0; i < tokens.length - 2; i++) {
            String a = tokens[i] + " " + tokens[i + 1];
            String b = tokens[i + 2];

            pairs.computeIfAbsent(a, k -> new HashMap<String, Integer>())
                    .merge(b, 1, Integer::sum);
            bigrams.computeIfAbsent(tokens[i + 1], k -> new HashMap<String, Integer>())
                    .merge(b, 1, Integer::sum);
        }

        String currentWord = "java is";
        StringBuilder sentence = new StringBuilder(currentWord);
        Random sentenceRandom = new Random();
        for (int step = 0; step < 30; step++) {
            Map<String, Integer> nextWords = pairs.get(currentWord);

            if(nextWords == null || nextWords.isEmpty()) {
                nextWords = bigrams.get(currentWord.split(" ")[1]);
            }

            if(nextWords == null || nextWords.isEmpty()) {
                break;
            }

            String[] words = sentence.toString().split(" ");
            Set<String> recent = new HashSet<>(Arrays.asList(words).subList(Math.max(0, words.length - 4), words.length));

            Map<String, Integer> fresh = new HashMap<>();
            for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
                if (!recent.contains(entry.getKey())) fresh.put(entry.getKey(), entry.getValue());
            }
            if (fresh.isEmpty()) fresh = nextWords;

            int total = 0;
            for (int count : fresh.values()) total += count;

            int roll = sentenceRandom.nextInt(total);
            String nextWord = null;
            for (Map.Entry<String, Integer> entry : fresh.entrySet()) {
                roll -= entry.getValue();
                if (roll < 0) {
                    nextWord = entry.getKey();
                    break;
                }
            }

            if (nextWord == null || "<end>".equals(nextWord)) {
                break;
            }

            sentence.append(" ").append(nextWord);

            String[] contextParts = currentWord.split(" ");
            currentWord = contextParts[1] + " " + nextWord;
        }

        System.out.println(sentence + ".");
    }
}
