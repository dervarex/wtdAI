package com.dervarex.wtdai;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("data.txt"));

        text = text.toLowerCase();
        text = text.replace(".", " <end> ");
        text = text.replace(",", "");
        text = text.replaceAll("\\[.*?\\]", "");
        text = text.replaceAll("[^a-z ]", " ");

        String word = "artificial";

        String[] tokens = text.split("\\s+");

        Map<String, Map<String, Integer>> pairs = new HashMap<>();

        for(int i = 0; i < tokens.length - 2; i++) {
            String a = tokens[i] + " " + tokens[i + 1];
            String b = tokens[i + 2];

            pairs.computeIfAbsent(a, k -> new HashMap<String, Integer>())
                    .merge(b, 1, Integer::sum);
        }

        Map<String, Integer> nextWords = pairs.getOrDefault(word, new HashMap<>());

        int totalOccurrences = 0;
        for (int count : nextWords.values()) {
            totalOccurrences += count;
        }
        System.out.println("Word: " + word + " | Total occurrences: " + totalOccurrences);

        for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
            String next = entry.getKey();
            int count = entry.getValue();
            double probability = (double) count / totalOccurrences;
            System.out.println(next + ": "+ (probability*100) + "%");
        }

        String bestOne = null;
        int highestCount =0;
        for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
            if(entry.getValue() > highestCount) {
                highestCount = entry.getValue();
                bestOne = entry.getKey();
            }
        }
        System.out.println("Best word after "+ word + ": " + bestOne);

        // sampling
        Random random = new Random();
        double randomValue = random.nextDouble();
        double comulative = 0;
        String chosen = null;

        for (Map.Entry<String, Integer> entry : nextWords.entrySet()) {
            double probability = (double) entry.getValue() / totalOccurrences;
            comulative += probability;
            if(randomValue < comulative) {
                chosen = entry.getKey();
                break;
            }
        }
        System.out.println("Chosen: " + chosen);

        // sentences
        String currentWord = "artificial intelligence";
        StringBuilder sentence = new StringBuilder(currentWord);
        Random sentenceRandom = new Random();
        for (int step = 0; step < 30; step++) {
            Map<String, Integer> nextWords2 = pairs.get(currentWord);

            if(nextWords2 == null || nextWords2.isEmpty()) {
                break;
            }

            int totalOccurences = 0;
            for (int count : nextWords2.values()) {
                totalOccurences += count;
            }

            double diceRoll = sentenceRandom.nextDouble();
            double runningTotal = 0;
            String nextWord = null;

            for (Map.Entry<String, Integer> entry : nextWords2.entrySet()) {
                double probability = (double) entry.getValue() / totalOccurences;
                runningTotal += probability;
                if(diceRoll < runningTotal) {
                    nextWord = entry.getKey();
                    break;
                }
            }

            if ("<end>".equals(nextWord)) {
                break;
            }

            sentence.append(" ").append(nextWord);

            String[] contextParts = currentWord.split(" ");
            currentWord = contextParts[1] + " " + nextWord;
        }

        System.out.println(sentence + ".");
    }
}
