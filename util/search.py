import torch
from org.apache.lucene.search import BooleanClause, BooleanQuery, BoostQuery
from ru.itmo.sparsifiermodel.query import FieldValueAsScoreQuery

from util.field import to_field_name


def build_query(query):
    builder = BooleanQuery.Builder()
    for term in torch.nonzero(query):
        field_name = to_field_name(term.item())
        value = query[term].item()
        builder.add(BoostQuery(FieldValueAsScoreQuery(field_name), value), BooleanClause.Occur.SHOULD)
    return builder.build()
