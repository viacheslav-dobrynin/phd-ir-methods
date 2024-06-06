package ru.itmo.sparsifiermodel.query

import org.apache.lucene.index.LeafReaderContext
import org.apache.lucene.search.DocIdSetIterator
import org.apache.lucene.search.Explanation
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.Query
import org.apache.lucene.search.QueryVisitor
import org.apache.lucene.search.ScoreMode
import org.apache.lucene.search.Scorer
import org.apache.lucene.search.Weight

class FieldValueAsScoreQuery(private val field: String) : Query() {

    override fun createWeight(searcher: IndexSearcher, scoreMode: ScoreMode, boost: Float): Weight =
        object : Weight(this) {
            override fun isCacheable(ctx: LeafReaderContext?): Boolean {
                TODO("Not yet implemented")
            }

            override fun explain(context: LeafReaderContext?, doc: Int): Explanation {
                TODO("Not yet implemented")
            }

            override fun scorer(context: LeafReaderContext): Scorer = object : Scorer(this) {
                val iterator = context.reader().getNumericDocValues(field)

                override fun score(): Float {
                    val docId = docID()
                    require(docId != DocIdSetIterator.NO_MORE_DOCS)
                    require(iterator.advanceExact(docId))
                    return Float.fromBits(iterator.longValue().toInt()) * boost
                }

                override fun docID(): Int = iterator.docID()
                override fun iterator(): DocIdSetIterator = iterator
                override fun getMaxScore(upTo: Int): Float = Float.MAX_VALUE
            }
        }

    override fun equals(other: Any?): Boolean {
        TODO("Not yet implemented")
    }


    override fun hashCode(): Int {
        val prime = 31
        var hash = classHash()
        hash = prime * hash + field.hashCode()
        return hash
    }

    override fun toString(field: String?): String = buildString {
        append("FieldValueAsScoreQuery [field=")
        append(this@FieldValueAsScoreQuery.field)
        append("]")
    }

    override fun visit(visitor: QueryVisitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this)
        }
    }
}
