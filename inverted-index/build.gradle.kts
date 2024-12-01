plugins {
    kotlin("jvm") version "2.1.0"
}

group = "ru.itmo.sparsifiermodel"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform(libs.djl.bom))
    implementation(libs.djl.pytorch.engine)
    implementation(libs.bundles.lucene)
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}
