package com.dervarex.wtdai;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        String text = "i drink water i drink juice i drink water";
        String word= "drink";

        String[] tokens = text.toLowerCase().split("\\s+");
        System.out.println("Tokens: " + Arrays.toString(tokens));

        Map<String, Map<String, Integer>> pairs = new HashMap<>();
        for(int i = 0; i < tokens.length -1; i++) {
            String a = tokens[i], b = tokens[i+1];
            pairs.computeIfAbsent(a, k -> new HashMap<String, Integer>())
                    .merge(b,1,Integer::sum);
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
    }
}
