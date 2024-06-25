package ru.itmo.sparsifiermodel

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document.Document
import org.apache.lucene.document.Field
import org.apache.lucene.document.FloatDocValuesField
import org.apache.lucene.document.StringField
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.search.BooleanClause
import org.apache.lucene.search.BooleanQuery
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.store.Directory
import org.apache.lucene.store.FSDirectory
import ru.itmo.sparsifiermodel.query.FieldValueAsScoreQuery
import java.nio.file.Files
import kotlin.system.measureTimeMillis

private val analyzer: Analyzer = StandardAnalyzer()
private val indexPath = Files.createTempDirectory("tempIndex")
private val directory: Directory = FSDirectory.open(indexPath)

fun main() {
    val indexTime = measureTimeMillis {
        NDManager.newBaseManager().use { manager ->
            val docs = manager.create(
                floatArrayOf(
                    0f, 3.5f, 0f,
                    4.5f, 0f, 5.5f,
                    0f, 6.5f, 0f
                ),
                Shape(3, 3)
            )
            buildIndex(docs)
        }
    }
    println("Index time: $indexTime ms")

    val searchTime = measureTimeMillis { search(floatArrayOf(1f, 2f, 0f)) }
    println("Search time: $searchTime ms")
}

fun buildIndex(embs: NDArray) {
    val config = IndexWriterConfig(analyzer)
    config.similarity = TodoSimilarity()
    val writer = IndexWriter(directory, config)
    for (i in 0 until embs.size(0)) {
        val emb = embs[i].toFloatArray()
        val doc = Document()
        doc.add(StringField("doc_id", "$i", Field.Store.YES))
        for ((j, value) in emb.withIndex()) if (value != 0f) {
            doc.add(FloatDocValuesField(j.toFieldName(), value))
        }
        println(doc)
        writer.addDocument(doc)
    }
    writer.close()
}

fun search(queryEmb: FloatArray) {
    val reader = DirectoryReader.open(FSDirectory.open(indexPath))
    val searcher = IndexSearcher(reader)
    searcher.similarity = TodoSimilarity()

    val hits = searcher.search(buildQuery(queryEmb), 10).scoreDocs
    println("Hits: ${hits.contentToString()}")

    // Iterate through the results:
    val storedFields = searcher.storedFields()
    for (i in hits.indices) {
        val hitDoc = storedFields.document(hits[i].doc)
        println("Found doc: $hitDoc. Score: ${hits[i].score}")
    }
    reader.close()
    directory.close()
}

fun buildQuery(query: FloatArray): BooleanQuery {
    val builder = BooleanQuery.Builder()
    for ((i, value) in query.withIndex()) if (value != 0f) {
        builder.add(FieldValueAsScoreQuery(i.toFieldName(), value), BooleanClause.Occur.SHOULD)
    }
    return builder.build()
}
