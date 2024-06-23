from org.apache.lucene.search import BooleanClause, BooleanQuery, BoostQuery
from ru.itmo.sparsifiermodel.query import FieldValueAsScoreQuery

from util.field import to_field_name


def build_query(query):
    builder = BooleanQuery.Builder()
    for i, value in enumerate(query): # TODO: change to torch.nonzero
        if value != 0.0:
            builder.add(BoostQuery(FieldValueAsScoreQuery(to_field_name(i)), value), BooleanClause.Occur.SHOULD)
    return builder.build()
