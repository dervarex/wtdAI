# wtdAI

while (true) doAI;

## Training data sources

### Wikipedia
Text from Wikipedia: citations, notices and non-ASCII characters were removed.

- `wikipedia-artificial_intelligence.txt`: https://en.wikipedia.org/wiki/Artificial_intelligence
- `wikipedia-git.txt`: https://en.wikipedia.org/wiki/Git
- `wikipedia-github.txt`: https://en.wikipedia.org/wiki/GitHub
- `wikipedia-java-programming-language.txt`: https://en.wikipedia.org/wiki/Java_(programming_language)

Wikipedia text is licensed under CC BY-SA 4.0.

### Gutenberg

- `gutenberg-the-notebooks-of-leonardo-da-vinci-complete.txt`: https://www.gutenberg.org/ebooks/5000
- `gutenberg-my-life-and-work.txt`: https://www.gutenberg.org/ebooks/7213
- `gutenberg-the-complete-works-of-william-shakespeare.txt`: https://www.gutenberg.org/ebooks/100

## Running

```
./gradlew run
./gradlew run --args="--debug"
```

Available arguments:

- `--debug`: turns on debug output

