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

    // TornadoVM for GPU kernels, from local dir
    val tornadoJars = file("tornadovm-install/share/java/tornado")
    if (tornadoJars.isDirectory) {
        implementation(fileTree(tornadoJars) {
            include("tornado-api-*.jar", "tornado-runtime-*.jar", "tornado-drivers-common-*.jar",
                    "tornado-drivers-cuda-*.jar", "tornado-drivers-opencl-*.jar", "tc-runtime-*.jar")
        })
    }
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "com.dervarex.wtdai.Main"
}
