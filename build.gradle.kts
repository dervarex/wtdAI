import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import javax.inject.Inject

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

// Training data
// ./gradlew downloadData                        downloads it from github releases
// ./gradlew downloadData -PdataUrl=file:///...  for a locally built archive
abstract class FetchData : DefaultTask() {
    @get:Input abstract val url: Property<String>
    @get:Input @get:Optional abstract val sha256: Property<String>
    @get:OutputFile abstract val archive: RegularFileProperty

    @TaskAction
    fun fetch() {
        val out = archive.get().asFile
        out.parentFile.mkdirs()
        val part = File(out.path + ".part")
        logger.lifecycle("downloading ${url.get()}")
        URI(url.get()).toURL().openStream().use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
        if (sha256.isPresent && sha256.get().isNotBlank()) {
            val digest = MessageDigest.getInstance("SHA-256")
            part.inputStream().use { input ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    digest.update(buf, 0, n)
                }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            if (!actual.equals(sha256.get().trim(), ignoreCase = true)) {
                part.delete()
                throw GradleException("training data checksum mismatch, expected ${sha256.get()} but got $actual")
            }
        }
        Files.move(part.toPath(), out.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }
}

val dataVersion = providers.gradleProperty("dataVersion")
val fetchData = tasks.register<FetchData>("fetchData") {
    description = "Downloads the training data archive into build/data"
    url = providers.gradleProperty("dataUrl").orElse(dataVersion.map {
        "https://github.com/dervarex/wtdAI/releases/download/data-$it/wtdai-data-$it.tar.gz"
    })
    sha256 = providers.gradleProperty("dataSha256") // -PdataSha256= skips the check
    archive = layout.buildDirectory.file(dataVersion.map { "data/wtdai-data-$it.tar.gz" })
}

interface ArchiveAccess {
    @get:Inject val archives: ArchiveOperations
}

tasks.register<Sync>("downloadData") {
    group = "wtdai"
    description = "Downloads and unpacks the training data into training-data/ (replaces whatever is there)"
    val archives = objects.newInstance<ArchiveAccess>().archives // tarTree() from the script breaks the configuration cache
    from(fetchData.flatMap { it.archive }.map { archives.tarTree(it) })
    into("training-data")
}
