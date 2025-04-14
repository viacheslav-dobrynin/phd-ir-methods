import torch

from common.pooling import mean_pooling


def build_encode_dense_fun(tokenizer, model, device):
    def encode_dense(docs):
        # Tokenize sentences
        encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

    return encode_dense
