plugins {
    application
}

group = "com.dervarex"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.dervarex.minified:minified-utils:4.0.4")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "com.dervarex.wtdai.Main"
}
