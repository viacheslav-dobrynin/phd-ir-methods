package ru.itmo.sparsifiermodel.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.*;

import java.io.IOException;
import java.util.Objects;

/**
 * Java is used so that this class can be used with PyLucene
 */
public class FieldValueAsScoreQuery extends Query {

    private final String field;

    public FieldValueAsScoreQuery(String field) {
        this.field = Objects.requireNonNull(field);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        return new Weight(this) {
            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                throw new UnsupportedOperationException();
            }

            @Override
            public Explanation explain(LeafReaderContext context, int doc) {
                throw new UnsupportedOperationException();
            }

            @Override
            public Scorer scorer(LeafReaderContext context) throws IOException {
                return new Scorer(this) {

                    private final NumericDocValues iterator = context.reader().getNumericDocValues(field);

                    @Override
                    public float score() throws IOException {
                        final int docId = docID();
                        assert docId != DocIdSetIterator.NO_MORE_DOCS;
                        assert iterator.advanceExact(docId);
                        return Float.intBitsToFloat((int) iterator.longValue()) * boost;
                    }

                    @Override
                    public int docID() {
                        return iterator.docID();
                    }

                    @Override
                    public DocIdSetIterator iterator() {
                        return iterator;
                    }

                    @Override
                    public float getMaxScore(int upTo) {
                        return Float.MAX_VALUE;
                    }
                };
            }
        };
    }

    @Override
    public String toString(String field) {
        return "FieldValueAsScoreQuery [field=" + this.field + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) && field.equals(((FieldValueAsScoreQuery) other).field);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int hash = classHash();
        hash = prime * hash + field.hashCode();
        return hash;
    }
}
