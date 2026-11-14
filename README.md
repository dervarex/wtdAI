# wtdAI

while (true) doAI;

## Training data sources

Text from Wikipedia, citations, notices and non-ASCII characters were removed.

- `wikipedia-artificial_intelligence.txt`: https://en.wikipedia.org/wiki/Artificial_intelligence
- `wikipedia-git.txt`: https://en.wikipedia.org/wiki/Git
- `wikipedia-github.txt`: https://en.wikipedia.org/wiki/GitHub
- `wikipedia-java-programming-language.txt`: https://en.wikipedia.org/wiki/Java_(programming_language)

Wikipedia text is licensed under CC BY-SA 4.0.

## Running

```
javac -d out src/main/java/com/dervarex/wtdai/Main.java
java -cp out com.dervarex.wtdai.Main
```
