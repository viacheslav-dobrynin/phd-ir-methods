package ru.itmo.sparsifiermodel.query;

import org.apache.lucene.index.DocValues;
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
    private final float queryTermValue;

    public FieldValueAsScoreQuery(String field, float queryTermValue) {
        this.field = Objects.requireNonNull(field);
        if (Float.isInfinite(queryTermValue) || Float.isNaN(queryTermValue)) {
            throw new IllegalArgumentException("Query term value must be finite and non-NaN");
        }
        this.queryTermValue = queryTermValue;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        return new Weight(this) {
            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return DocValues.isCacheable(ctx, field);
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
                        return Float.intBitsToFloat((int) iterator.longValue()) * queryTermValue * boost;
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
        StringBuilder builder = new StringBuilder();
        builder.append("FieldValueAsScoreQuery [field=");
        builder.append(this.field);
        builder.append(", queryTermValue=");
        builder.append(this.queryTermValue);
        builder.append("]");
        return builder.toString();
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) && equalsTo(getClass().cast(other));
    }

    private boolean equalsTo(FieldValueAsScoreQuery other) {
        return field.equals(other.field)
                && Float.floatToIntBits(queryTermValue) == Float.floatToIntBits(other.queryTermValue);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int hash = classHash();
        hash = prime * hash + field.hashCode();
        hash = prime * hash + Float.floatToIntBits(queryTermValue);
        return hash;
    }
}
