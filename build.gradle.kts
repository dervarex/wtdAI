plugins {
    application
}

group = "com.dervarex"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {

}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "com.dervarex.wtdai.Main"
}
