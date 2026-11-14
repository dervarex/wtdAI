package com.dervarex.wtdai;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        String text = "i drink water i drink juice i drink water";

        String[] tokens = text.toLowerCase().split("\\s+");
        System.out.println("Tokens: " + Arrays.toString(tokens));
    }
}
