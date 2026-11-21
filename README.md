# wtdAI

while (true) doAI;

A small neural language model written from scratch in Java

## How it works

1. **Tokenizer** reads the training data, lowercases it and cuts it into tokens
2. **Vocabulary** keeps the most common words. Everything else becomes `<unk>`.
3. **LLM** learns the stuff.
4. **Generator** starts from a seed such as `java is` and makes one word at a time until it reaches a period or the word limit.

## Training data

The training data isn't anymore in this repo, it's about 50 million words in thousands of files, indexing it would nuke your computer :(
You gotta download it from [GitHub releases](https://github.com/dervarex/wtdAI/releases) (tag `data-v1` and so on):
For that, use the below command:
```
./gradlew downloadData
```

It'll download that archive into `build/data/`, check the hash and unpack it into `training-data/`.
Use `--data` for a different folder while training, if you want to try it out with less (or more) training data.

The archive is built from the Wikipedia dumps.
`manifest.jsonl` lists the source page and when I got it.

### Copyright and licenses

Training data comes from Wikipedia, English and simple english.
Wikipedia text is licensed under the [Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
See `manifest.jsonl` in the training data for links and dates of downloading.

Changes:
- Templates, tables, references, lists, headings, images and markup were removed.
- Non-ASCII characters were converted or removed, digits and punctuation besides periods and apostrophes were stripped.

## Running

Build and run with Gradle from the project root, download the training data first:

```
./gradlew downloadData
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
| `--save PATH`    | off       | Writes the trained model after every epoch                |
| `--load PATH`    | off       | Skips training and loads the model                        |
| `--resume PATH`  | off       | Keeps training a saved model up to `--epochs`|
| `--data PATH`    | `training-data` | Folder with the training data `.txt` files          |
| `--max-words N`  | all       | Only reads files until it has this many words             |

### Training on the GPU

You'll need [TornadoVM](https://github.com/beehive-lab/TornadoVM)

```
./gradlew downloadTornado
```

Then:

```
./gradlew runGpu --args="--debug --epochs 2 --save model.bin"
```

On an RTX 3060 one epoch over all 50 million words takes about 15 minutes, depending on your GPU it can take longer or shorter.

### Running it on your pc without training

I'll release models alongside the source code on GitHub releases, you can load them like that:

CPU:
```
./gradlew run --args="--load model.bin --sentences 5"
```

GPU:
```
./gradlew runGpu --args="--load model.bin --sentences 5"
```