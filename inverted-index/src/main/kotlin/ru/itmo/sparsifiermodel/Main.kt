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
import org.apache.lucene.expressions.SimpleBindings
import org.apache.lucene.expressions.js.JavascriptCompiler
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.queries.function.FunctionQuery
import org.apache.lucene.queries.function.FunctionScoreQuery
import org.apache.lucene.queries.function.ValueSource
import org.apache.lucene.queries.function.valuesource.ConstValueSource
import org.apache.lucene.queries.function.valuesource.FloatFieldSource
import org.apache.lucene.queries.function.valuesource.ProductFloatFunction
import org.apache.lucene.queries.function.valuesource.SumFloatFunction
import org.apache.lucene.search.BooleanClause
import org.apache.lucene.search.BooleanQuery
import org.apache.lucene.search.DoubleValuesSource
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.MatchAllDocsQuery
import org.apache.lucene.search.Query
import org.apache.lucene.store.Directory
import org.apache.lucene.store.FSDirectory
import ru.itmo.sparsifiermodel.query.FieldValueAsScoreQuery
import java.nio.file.Files
import kotlin.system.measureTimeMillis

private val analyzer: Analyzer = StandardAnalyzer()
private val indexPath = Files.createTempDirectory("tempIndex")
private val directory: Directory = FSDirectory.open(indexPath)

fun main() {
    var indexTime = measureTimeMillis {
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

    val queryEmb = floatArrayOf(1f, 2f, 0f)

    var searchTime = measureTimeMillis { search(buildBooleanQuery(queryEmb)) }
    println("Search time (custom): $searchTime ms")
    searchTime = measureTimeMillis { search(buildFunctionQuery(queryEmb)) }
    println("Search time (FQ): $searchTime ms")
    searchTime = measureTimeMillis { search(buildQueryViaExpression(queryEmb)) }
    println("Search time (exp): $searchTime ms")


    println("==Big vectors example==")
    indexTime = measureTimeMillis {
        NDManager.newBaseManager().use { manager ->
            val docs = manager.randomNormal(Shape(1, 768))
            buildIndex(docs)
        }
    }
    println("Index time: $indexTime ms")

    searchTime = measureTimeMillis {
        NDManager.newBaseManager().use { manager ->
            search(buildBooleanQuery(manager.randomNormal(Shape(1, 768)).toFloatArray()))
        }
    }
    println("Search time: $searchTime ms")

    directory.close()
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

private fun buildBooleanQuery(query: FloatArray): BooleanQuery {
    val builder = BooleanQuery.Builder()
    for ((i, value) in query.withIndex()) if (value != 0f) {
        builder.add(FieldValueAsScoreQuery(i.toFieldName(), value), BooleanClause.Occur.SHOULD)
    }
    return builder.build()
}

private fun buildFunctionQuery(queryEmb: FloatArray): FunctionQuery {
    val productFunctions = mutableListOf<ValueSource>()
    for ((i, qi) in queryEmb.withIndex()) if (qi != 0f) {
        productFunctions += ProductFloatFunction(arrayOf(ConstValueSource(qi), FloatFieldSource(i.toFieldName())))
    }
    val dotProductQuery = FunctionQuery(SumFloatFunction(productFunctions.toTypedArray()))
    return dotProductQuery
}

private fun buildQueryViaExpression(queryEmb: FloatArray): FunctionScoreQuery {
    val expressionBuilder = StringBuilder()
    val bindings = SimpleBindings()
    for ((i, qi) in queryEmb.withIndex()) if (qi != 0f) {
        if (expressionBuilder.isNotBlank()) expressionBuilder.append(" + ")
        expressionBuilder.append(qi).append(" * ").append(i.toFieldName())
        bindings.add(i.toFieldName(), DoubleValuesSource.fromFloatField(i.toFieldName()))
    }
    val dotProductExpression = JavascriptCompiler.compile(expressionBuilder.toString())
    val dotProductQuery = FunctionScoreQuery(MatchAllDocsQuery(), dotProductExpression.getDoubleValuesSource(bindings))
    return dotProductQuery
}

private fun search(query: Query) {
    val reader = DirectoryReader.open(FSDirectory.open(indexPath))
    val searcher = IndexSearcher(reader)
    searcher.similarity = TodoSimilarity()

    val hits = searcher.search(query, 10).scoreDocs
    println("Hits: ${hits.contentToString()}")

    // Iterate through the results:
    val storedFields = searcher.storedFields()
    for (i in hits.indices) {
        val hitDoc = storedFields.document(hits[i].doc)
        println("Found doc: $hitDoc. Score: ${hits[i].score}")
    }
    reader.close()
}
