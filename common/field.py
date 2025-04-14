from org.apache.lucene.document import StringField, Field


def to_field_name(i):
    return f"term_{i}"


def to_doc_id_field(doc_id):
    return StringField("doc_id", str(doc_id), Field.Store.YES)
