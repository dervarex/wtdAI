package com.dervarex.wtdai;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        String text = "i drink water i drink juice i like water i like juice " +
                "you drink water you drink juice you like water you like juice " +
                "we drink water we like juice we drink coffee " +
                "i drink coffee i like coffee you like coffee " +
                "the cat drinks water the cat likes juice " +
                "the dog drinks water the dog likes bones " +
                "i eat bread i eat apples you eat bread " +
                "we eat apples and bread " +
                "the cat eats fish the dog eats bones " +
                "i like the cat i like the dog " +
                "the cat drinks water and eats fish " +
                "the dog drinks water and eats bones " +
                "you and i drink water " +
                "we like the cat and the dog " +
                "i drink water every day " +
                "you drink juice every day " +
                "the cat sleeps the dog sleeps " +
                "i sleep you sleep we sleep " +
                "i like water and juice " +
                "you like coffee and bread";
        String word= "drink";

        String[] tokens = text.toLowerCase().split("\\s+");
        System.out.println("Tokens: " + Arrays.toString(tokens));

        Map<String, Map<String, Integer>> pairs = new HashMap<>();

        for(int i = 0; i < tokens.length - 2; i++) {
            String a = tokens[i] + " " + tokens[i + 1];
            String b = tokens[i + 2];

            pairs.computeIfAbsent(a, k -> new HashMap<String, Integer>())
                    .merge(b, 1, Integer::sum);
        }

        System.out.println("Pairs: " + pairs);

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
        String currentWord = "i drink";

        StringBuilder sentence = new StringBuilder(currentWord);

        Random sentenceRandom = new Random();

        for (int step = 0; step < 10; step++) {
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
            sentence.append(" ").append(nextWord);

            String[] contextParts = currentWord.split(" ");
            currentWord = contextParts[1] + " " + nextWord;
        }
        System.out.println(sentence);
    }
}
