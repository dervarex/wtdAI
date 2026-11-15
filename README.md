# wtdAI

while (true) doAI;

A small neural language model written from scratch in Java

## How it works

1. **Tokenizer** reads the training data, lowercases it and cuts it into tokens
2. **Vocabulary** keeps the most common words. Everything else becomes `<unk>`.
3. **LLM** learns the stuff.
4. **Generator** starts from a seed such as `java is` and makes one word at a time until it reaches a period or the word limit.

## Training data

See [training-data.jsonl](training-data.jsonl)

### Copyright and licenses

Training data comes from Wikipedia, English and simple english.
Wikipedia text is licensed under the [Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
See [training-data.jsonl](training-data.jsonl) for links and dates.

Changes:
- Citations, notices, templates and markup were removed.
- Non-ASCII characters were removed, and punctuation was stripped.

## Running

Build and run with Gradle from the project root:

```
./gradlew run
./gradlew run --args="--debug --epochs 4 --sentences 5 --seed 'java is'"
```

Arguments use the `--` prefix from `minified-utils`'s `Parser`:

| Argument         | Default   | What it does                                              |
|------------------|-----------|-----------------------------------------------------------|
| `--debug`        | off       | Prints token and vocabulary counts and the loss per epoch |
| `--epochs N`     | 2         | Passes over the training text                             |
| `--emb N`        | 16        | Size of each word embedding                               |
| `--hidden N`     | 64        | Size of the hidden layer                                  |
| `--lr X`         | 0.05      | Learning rate                                             |
| `--sentences N`  | 3         | How many sentences to generate                            |
| `--seed "words"` | `java is` | The words the sentence starts with                        |
| `--save PATH`    | off       | Writes the trained model after training                   |
| `--load PATH`    | off       | Skips training and loads the model                        |

### Running it on your pc without training

I'll release models alongside the source code on GitHub releases, you can load them like that:

```
./gradlew run --args="--load model.bin --sentences 5"
```


