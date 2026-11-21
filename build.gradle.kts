import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import groovy.json.JsonSlurper
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
    fun fetch() = Download.fetch(url.get(), archive.get().asFile, sha256.orNull, "training data")
}

object Download {
    fun fetch(url: String, out: File, sha256: String?, what: String) {
        out.parentFile.mkdirs()
        val part = File(out.path + ".part")
        println("downloading $url")
        URI(url).toURL().openStream().use { input -> part.outputStream().use { input.copyTo(it, 1 shl 20) } }
        if (!sha256.isNullOrBlank()) {
            val actual = sha256Of(part)
            if (!actual.equals(sha256.trim(), ignoreCase = true)) {
                part.delete()
                throw GradleException("$what checksum mismatch, expected $sha256 but got $actual")
            }
        }
        Files.move(part.toPath(), out.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }

    fun sha256Of(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

val dataVersion = providers.gradleProperty("dataVersion")
val fetchData = tasks.register<FetchData>("fetchData") {
    description = "Downloads the training data archive into build/data"
    url = providers.gradleProperty("dataUrl").orElse(dataVersion.map {
        "https://github.com/dervarex/wtdAI/releases/download/data-$it/wtdai-data-$it.tar.gz"
    })
    sha256 = providers.gradleProperty("dataSha256")
    archive = layout.buildDirectory.file(dataVersion.map { "data/wtdai-data-$it.tar.gz" })
}

interface ArchiveAccess {
    @get:Inject val archives: ArchiveOperations
}

tasks.register<Sync>("downloadData") {
    group = "wtdai"
    description = "Downloads and unpacks the training data into training-data/ (replaces whatever is there)"
    val archives = objects.newInstance<ArchiveAccess>().archives
    from(fetchData.flatMap { it.archive }.map { archives.tarTree(it) })
    into("training-data")
}

abstract class InstallTornado @Inject constructor(
    private val fs: FileSystemOperations,
    private val archives: ArchiveOperations,
) : DefaultTask() {
    @get:Input abstract val version: Property<String>
    @get:Input abstract val backend: Property<String>
    @get:Input abstract val platform: Property<String>
    @get:Input @get:Optional abstract val sha256: Property<String>
    @get:Input @get:Optional abstract val nvrtcRelease: Property<String>
    @get:Input abstract val javaHome: Property<String>
    @get:Internal abstract val downloads: DirectoryProperty
    @get:OutputDirectory abstract val installDir: DirectoryProperty

    private val windows get() = platform.get().startsWith("windows")

    @TaskAction
    fun install() {
        val home = installDir.get().asFile
        fs.delete { delete(home) }
        home.mkdirs()

        val asset = "tornadovm-${version.get()}-jdk22plus-${backend.get()}-${platform.get()}.tar.gz"
        val sdk = cached(
            "https://github.com/beehive-lab/TornadoVM/releases/download/v${version.get()}/$asset",
            asset, sha256.orNull ?: TORNADO_SHA256[asset]
                ?: throw GradleException("no known checksum for $asset, pass -PtornadoSha256=<sha256 from the release page>"))
        fs.copy {
            from(archives.tarTree(sdk))
            eachFile { relativePath = RelativePath(!isDirectory, *relativePath.segments.drop(1).toTypedArray()) }
            includeEmptyDirs = false
            into(home)
        }
        File(home, "bin").listFiles()?.forEach { it.setExecutable(true) }

        val needsNvrtc = backend.get() == "cuda" || backend.get() == "full"
        if (needsNvrtc) installNvrtc(File(home, "nvrtc"))
        writeEnv(home, needsNvrtc)
        check(home, needsNvrtc)
    }

    private fun cached(url: String, name: String, sha: String): File {
        val file = downloads.file(name).get().asFile
        if (!file.isFile || !Download.sha256Of(file).equals(sha, ignoreCase = true)) Download.fetch(url, file, sha, name)
        return file
    }

    private fun installNvrtc(dir: File) {
        val release = nvrtcRelease.orNull?.takeIf { it.isNotBlank() } ?: nvrtcForDriver()
        val nvPlatform = when (platform.get()) {
            "linux-amd64" -> "linux-x86_64"
            "linux-aarch64" -> "linux-sbsa"
            "windows-amd64" -> "windows-x86_64"
            else -> throw GradleException("no NVRTC for ${platform.get()}, try -PtornadoBackend=opencl")
        }
        val base = "https://developer.download.nvidia.com/compute/cuda/redist/"
        @Suppress("UNCHECKED_CAST")
        val index = JsonSlurper().parse(URI("${base}redistrib_$release.json").toURL()) as Map<String, Any>
        @Suppress("UNCHECKED_CAST")
        val entry = (index["cuda_nvrtc"] as Map<String, Any>)[nvPlatform] as Map<String, String>
        val path = entry.getValue("relative_path")
        println("NVRTC from CUDA $release")
        val archive = cached(base + path, path.substringAfterLast('/'), entry.getValue("sha256"))

        if (archive.name.endsWith(".zip")) {
            fs.copy {
                from(archives.zipTree(archive))
                eachFile { relativePath = RelativePath(!isDirectory, *relativePath.segments.drop(1).toTypedArray()) }
                includeEmptyDirs = false
                into(dir)
            }
        } else {
            dir.mkdirs()
            val tar = ProcessBuilder("tar", "-xJf", archive.absolutePath, "-C", dir.absolutePath, "--strip-components=1")
                .inheritIO().start().waitFor()
            if (tar != 0) throw GradleException("unpacking ${archive.name} failed, is xz installed?")
        }
    }

    private fun nvrtcForDriver(): String {
        val cuda = try {
            val p = ProcessBuilder("nvidia-smi").redirectErrorStream(true).start()
            val out = p.inputStream.bufferedReader().readText()
            p.waitFor()
            Regex("CUDA Version: (\\d+)").find(out)?.groupValues?.get(1)?.toInt()
        } catch (e: Exception) {
            null
        }
        return when {
            cuda == null -> {
                logger.warn("nvidia-smi didn't work, is the NVIDIA driver installed? using NVRTC from CUDA 13")
                "13.0.2"
            }
            cuda >= 13 -> "13.0.2"
            else -> "12.4.1"
        }
    }

    private fun writeEnv(home: File, nvrtc: Boolean) {
        val java = javaHome.get()
        val nixDriver = File("/run/opengl-driver/lib").takeIf { it.isDirectory }
        File(home, "env.sh").writeText(buildString {
            appendLine("# made by ./gradlew downloadTornado, use it with: source tornadovm-install/env.sh")
            appendLine("export JAVA_HOME=\"$java\"")
            appendLine("export TORNADOVM_HOME=\"${home.absolutePath}\"")
            appendLine("export PATH=\"\$TORNADOVM_HOME/bin:\$JAVA_HOME/bin:\$PATH\"")
            val libs = mutableListOf<String>()
            if (nvrtc) {
                appendLine("export CUDA_HOME=\"\$TORNADOVM_HOME/nvrtc\"")
                libs += "\$CUDA_HOME/lib"
            }
            if (nixDriver != null) libs += nixDriver.path
            if (libs.isNotEmpty()) appendLine("export LD_LIBRARY_PATH=\"${libs.joinToString(":")}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"")
        })
        File(home, "env.cmd").writeText(buildString {
            appendLine("@rem made by gradlew downloadTornado, run it in cmd before using tornado")
            appendLine("set \"JAVA_HOME=$java\"")
            appendLine("set \"TORNADOVM_HOME=${home.absolutePath}\"")
            if (nvrtc) appendLine("set \"CUDA_PATH=%TORNADOVM_HOME%\\nvrtc\"")
            appendLine("set \"PATH=%TORNADOVM_HOME%\\bin;${if (nvrtc) "%CUDA_PATH%\\bin;" else ""}%JAVA_HOME%\\bin;%PATH%\"")
        })
    }

    private fun check(home: File, nvrtc: Boolean) {
        if (windows) {
            println("installed into ${home.path}, run tornadovm-install\\env.cmd and then: tornado --devices")
            return
        }
        val python = System.getenv("PATH").orEmpty().split(File.pathSeparator).any { File(it, "python3").canExecute() }
        if (!python) logger.warn("python3 is missing, the tornado launcher needs it, install it with your package manager")
        val pb = ProcessBuilder("bash", "-c", "source '${home.absolutePath}/env.sh' && tornado --devices")
            .redirectErrorStream(true)
        val p = pb.start()
        val out = p.inputStream.bufferedReader().readText()
        val ok = p.waitFor() == 0 && out.contains("Tornado device=")
        println(out.lines().filter { it.isNotBlank() && !it.trimStart().startsWith("at ") }.take(20).joinToString("\n"))
        if (ok) {
            println("TornadoVM works. Use it with: source tornadovm-install/env.sh")
        } else {
            logger.warn("TornadoVM is installed but doesn't see a device. Is the GPU driver installed? " +
                if (nvrtc) "For an AMD or Intel GPU use -PtornadoBackend=opencl." else "")
        }
    }

    companion object {
        val TORNADO_SHA256 = mapOf(
            "tornadovm-8.1.0-jdk22plus-cuda-linux-aarch64.tar.gz" to "336b7230172dacbe9af9354f69e20ac4c5fea574ce686b9ad954f76c33f8064a",
            "tornadovm-8.1.0-jdk22plus-cuda-linux-amd64.tar.gz" to "fe8871f33f28e21d0c85f6ea2cca62a69f80caab5b2c1335a8d94b4428a5a9b0",
            "tornadovm-8.1.0-jdk22plus-cuda-windows-amd64.tar.gz" to "bcf138ec0c7bda334e0a0e1d9867522d74ffef303f77ab8807487205aa5c3eb4",
            "tornadovm-8.1.0-jdk22plus-full-linux-amd64.tar.gz" to "7d8da3066530af05f490d3d11408c11f59ccb6cb4f766b660d4a1f6ed1ca8cbb",
            "tornadovm-8.1.0-jdk22plus-metal-mac-aarch64.tar.gz" to "4b116a3f531c643b366d7d03de051f5fcec30a0aa9005f34534be35a6fcc1bd2",
            "tornadovm-8.1.0-jdk22plus-opencl-linux-amd64.tar.gz" to "19d0a4723bbb1b825f8dc4dbd099b461042eb708e76df3cd2321b0dc98cd0087",
            "tornadovm-8.1.0-jdk22plus-opencl-mac-aarch64.tar.gz" to "5d9a537a3fec11717238861cb9e93a4e30381eb652599c8f0af9ed8a5aa2de8c",
            "tornadovm-8.1.0-jdk22plus-opencl-windows-amd64.tar.gz" to "46bdd3f901f72a9b0fca6c1e6e16b28e284bf3c76514e018566cc494d2299fee",
        )
    }
}

val tornadoPlatform = System.getProperty("os.name").lowercase().let { os ->
    when {
        os.contains("win") -> "windows-amd64"
        os.contains("mac") -> "mac-aarch64"
        System.getProperty("os.arch") == "aarch64" -> "linux-aarch64"
        else -> "linux-amd64"
    }
}

val nvidia =file("/proc/driver/nvidia/version").exists() ||
        file("${System.getenv("SystemRoot") ?: "C:\\Windows"}\\System32\\nvidia-smi.exe").exists()
val defaultBackend = when {
    tornadoPlatform.startsWith("mac") -> "metal"
    nvidia -> "cuda"
    else -> "opencl"
}

tasks.register<InstallTornado>("downloadTornado") {
    group = "wtdai"
    description = "Downloads TornadoVM (and NVRTC for CUDA) into tornadovm-install/ for --gpu"
    version = providers.gradleProperty("tornadoVersion")
    backend = providers.gradleProperty("tornadoBackend").orElse(defaultBackend)
    platform = tornadoPlatform
    sha256 = providers.gradleProperty("tornadoSha256")
    nvrtcRelease = providers.gradleProperty("nvrtcRelease")
    javaHome = javaToolchains.launcherFor(java.toolchain).map { it.metadata.installationPath.asFile.absolutePath }
    downloads = layout.buildDirectory.dir("tornado")
    installDir = layout.projectDirectory.dir("tornadovm-install")
}

abstract class RunGpu @Inject constructor(private val execOps: ExecOperations) : DefaultTask() {
    @get:Classpath abstract val classpath: ConfigurableFileCollection
    @get:Input @get:Option(option = "args", description = "Arguments for wtdAI, same as for run") abstract val appArgs: Property<String>
    @get:Internal abstract val tornadoHome: DirectoryProperty
    @get:Internal abstract val workDir: DirectoryProperty

    @TaskAction
    fun runOnGpu() {
        val env = tornadoHome.file("env.sh").get().asFile
        if (!env.isFile) throw GradleException("TornadoVM is missing, run ./gradlew downloadTornado first")
        val cp = classpath.files.filterNot { it.name.startsWith("tornado-") || it.name.startsWith("tc-") }
            .joinToString(File.pathSeparator) { it.absolutePath }
        execOps.exec {
            workingDir = this@RunGpu.workDir.get().asFile
            commandLine("bash", "-c", "source \"\$1\" && exec tornado --classpath \"\$2\" --params=\"\$3\" com.dervarex.wtdai.Main",
                "bash", env.absolutePath, cp, "--gpu --batch 256 " + appArgs.get())
        }
    }
}

tasks.register<RunGpu>("runGpu") {
    group = "wtdai"
    description = "Runs wtdAI on the GPU with the TornadoVM launcher, needs ./gradlew downloadTornado"
    classpath.from(sourceSets.main.map { it.runtimeClasspath })
    appArgs.convention("")
    tornadoHome = layout.projectDirectory.dir("tornadovm-install")
    workDir = layout.projectDirectory
}
