package ru.itmo.sparsifiermodel

import org.apache.lucene.index.FieldInvertState
import org.apache.lucene.search.CollectionStatistics
import org.apache.lucene.search.TermStatistics
import org.apache.lucene.search.similarities.Similarity

class TodoSimilarity : Similarity() {
    override fun computeNorm(state: FieldInvertState?): Long {
        TODO("Not yet implemented")
    }

    override fun scorer(
        boost: Float,
        collectionStats: CollectionStatistics?,
        vararg termStats: TermStatistics?,
    ): SimScorer {
        TODO("Not yet implemented")
    }
}
