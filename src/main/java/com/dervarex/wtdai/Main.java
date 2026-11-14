package com.dervarex.wtdai;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        String text = "i drink water i drink juice i drink water";

        String[] tokens = text.toLowerCase().split("\\s+");
        System.out.println("Tokens: " + Arrays.toString(tokens));

        Map<String, Map<String, Integer>> pairs = new HashMap<>();
        for(int i = 0; i < tokens.length -1; i++) {
            String a = tokens[i], b = tokens[i+1];
            pairs.computeIfAbsent(a, k -> new HashMap<String, Integer>())
                    .merge(b,1,Integer::sum);
        }
        System.out.println("Pairs: " + pairs);
    }
}
