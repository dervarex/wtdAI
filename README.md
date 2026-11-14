# wtdAI

while (true) doAI;

A small neural language model written from scratch in Java

## How it works

1. **Tokenizer** reads the training data, lowercases it and cuts it into tokens
2. **Vocabulary** keeps the most common words. Everything else becomes `<unk>`.
3. **LLM** learns the stuff.
4. **Generator** starts from a seed such as `java is` and makes one word at a time until it reaches a period or the word limit.

## Training data

The training text is about 189k words of Wikipedia, programming, Linux and other tech. So the model is gonna speak more UNIX than English.
Wikipedia citations, notices and non-ASCII characters were removed, and wrapped lines were joined into paragraphs (I don't know if you have to mention that).

Sources (all from Wikipedia, fetched 14 Nov 2026):

- `wikipedia-linux.txt`: https://en.wikipedia.org/wiki/Linux
- `wikipedia-linux-kernel.txt`: https://en.wikipedia.org/wiki/Linux_kernel
- `wikipedia-linux-distribution.txt`: https://en.wikipedia.org/wiki/Linux_distribution
- `wikipedia-debian.txt`: https://en.wikipedia.org/wiki/Debian
- `wikipedia-arch-linux.txt`: https://en.wikipedia.org/wiki/Arch_Linux
- `wikipedia-ubuntu.txt`: https://en.wikipedia.org/wiki/Ubuntu
- `wikipedia-unix.txt`: https://en.wikipedia.org/wiki/Unix
- `wikipedia-unix-philosophy.txt`: https://en.wikipedia.org/wiki/Unix_philosophy
- `wikipedia-operating-system.txt`: https://en.wikipedia.org/wiki/Operating_system
- `wikipedia-kernel-operating-system.txt`: https://en.wikipedia.org/wiki/Kernel_(operating_system)
- `wikipedia-shell-computing.txt`: https://en.wikipedia.org/wiki/Shell_(computing)
- `wikipedia-bash-unix-shell.txt`: https://en.wikipedia.org/wiki/Bash_(Unix_shell)
- `wikipedia-command-line-interface.txt`: https://en.wikipedia.org/wiki/Command-line_interface
- `wikipedia-terminal-emulator.txt`: https://en.wikipedia.org/wiki/Terminal_emulator
- `wikipedia-package-manager.txt`: https://en.wikipedia.org/wiki/Package_manager
- `wikipedia-free-and-open-source-software.txt`: https://en.wikipedia.org/wiki/Free_and_open-source_software
- `wikipedia-open-source-software.txt`: https://en.wikipedia.org/wiki/Open-source_software
- `wikipedia-git.txt`: https://en.wikipedia.org/wiki/Git
- `wikipedia-github.txt`: https://en.wikipedia.org/wiki/GitHub
- `wikipedia-version-control.txt`: https://en.wikipedia.org/wiki/Version_control
- `wikipedia-java-programming-language.txt`: https://en.wikipedia.org/wiki/Java_(programming_language)
- `wikipedia-java-virtual-machine.txt`: https://en.wikipedia.org/wiki/Java_virtual_machine
- `wikipedia-object-oriented-programming.txt`: https://en.wikipedia.org/wiki/Object-oriented_programming
- `wikipedia-compiler.txt`: https://en.wikipedia.org/wiki/Compiler
- `wikipedia-programming-language.txt`: https://en.wikipedia.org/wiki/Programming_language
- `wikipedia-computer-programming.txt`: https://en.wikipedia.org/wiki/Computer_programming
- `wikipedia-software-development.txt`: https://en.wikipedia.org/wiki/Software_development
- `wikipedia-source-code.txt`: https://en.wikipedia.org/wiki/Source_code
- `wikipedia-algorithm.txt`: https://en.wikipedia.org/wiki/Algorithm
- `wikipedia-data-structure.txt`: https://en.wikipedia.org/wiki/Data_structure
- `wikipedia-text-editor.txt`: https://en.wikipedia.org/wiki/Text_editor
- `wikipedia-vim-text-editor.txt`: https://en.wikipedia.org/wiki/Vim_(text_editor)
- `wikipedia-emacs.txt`: https://en.wikipedia.org/wiki/Emacs
- `wikipedia-regular-expression.txt`: https://en.wikipedia.org/wiki/Regular_expression
- `wikipedia-computer-file.txt`: https://en.wikipedia.org/wiki/Computer_file
- `wikipedia-software-bug.txt`: https://en.wikipedia.org/wiki/Software_bug
- `wikipedia-web-browser.txt`: https://en.wikipedia.org/wiki/Web_browser
- `wikipedia-computer-network.txt`: https://en.wikipedia.org/wiki/Computer_network
- `wikipedia-internet.txt`: https://en.wikipedia.org/wiki/Internet

Wikipedia text is licensed under CC BY-SA 4.0.

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


