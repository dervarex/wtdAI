/**
 * <h2>wtdAI - while(true) doAI;</h2>
 * <br><br>
 * A (small) neural network written in java, from scratch.
 * <br>
 * <h3>{@link LLM}</h3>
 * The Entrypoint and Core of the Neural Network.
 * That's where the training happens, the kinda complicated part.
 * <h3>{@link Generator}</h3>
 * After Training, we want it to generate sentences, that's what this class is for.
 * <h3>{@link ModelIO}</h3>
 * Saving and Loading of our model
 * <h3>{@link Tokenizer}</h3>
 * Finds the training data files and cuts text into Tokens
 * <h3>{@link Corpus}</h3>
 * Reads all training data files and turns them into word IDs, split into training data and test stuff
 * <h3>{@link Vocabulary}</h3>
 * Turns the words from the Training data into numbers
 *
 */
package com.dervarex.wtdai;