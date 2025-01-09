from collections import defaultdict

import argparse
import numpy as np
import faiss
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

from dataset import load_dataset
from params import DEVICE
from util.iterable import chunked
import pickle
import json
import lmdb

# (1012, '.', 8327750), (1996, 'the', 6514526), (1010, ',', 5931143), (1997, 'of', 3529956), (1037, 'a', 3332838),
# (1998, 'and', 3025636), (2000, 'to', 2786343), (1999, 'in', 2263387), (2003, 'is', 2125424), (101, '[CLS]', 2038782),
# (102, '[SEP]', 2038782), (1011, '-', 1465785), (2005, 'for', 1235799), (2030, 'or', 1069425), (1007, ')', 1046361),
# (1006, '(', 1032016), (2017, 'you', 932744), (2008, 'that', 886083), (2024, 'are', 849999), (2015, '##s', 819222),
# (2004, 'as', 794620), (1024, ':', 790401), (2009, 'it', 790204), (2006, 'on', 774770), (2007, 'with', 734534),
# (1005, "'", 702477), (2115, 'your', 681225), (2019, 'an', 575466), (2022, 'be', 563128), (2011, 'by', 560066),
# (1015, '1', 556094), (2013, 'from', 553582), (2064, 'can', 547802), (1055, 's', 488062), (2023, 'this', 481237),
# (2012, 'at', 440803), (1016, '2', 437881), (2031, 'have', 383660), (1013, '/', 350300), (2025, 'not', 347055),
# (2065, 'if', 335428), (1002, '$', 324706), (1017, '3', 301697), (2050, '##a', 298746), (1045, 'i', 298045),
# (2097, 'will', 292874), (2028, 'one', 288997), (2029, 'which', 277397), (2062, 'more', 268143), (2001, 'was', 266342),
# (2038, 'has', 255788), (2035, 'all', 254221), (1025, ';', 243999), (2021, 'but', 237943), (2089, 'may', 237306),
# (2036, 'also', 237153), (2043, 'when', 230894), (2027, 'they', 227915), (2060, 'other', 223216), (2087, 'most', 220996),
# (2055, 'about', 201142), (1018, '4', 192022), (3022, '##as', 189030), (2039, 'up', 185993), (2045, 'there', 185275),
# (2084, 'than', 184336), (1019, '5', 178267), (2037, 'their', 168207),
stop_tokens = {1012, 1996, 1010, 1997, 1037, 1998, 2000, 1999, 2003, 101, 102, 1011, 2005, 2030, 1007, 1006, 2017, 2008,
               2024, 2015, 2004, 1024, 2009, 2006, 2007, 1005, 2115, 2019, 2022, 2011, 1015, 2013, 2064, 1055, 2023,
               2012, 1016, 2031, 1013, 2025, 2065, 1002, 1017, 2050, 1045, 2097, 2028, 2029, 2062, 2001, 2038, 2035,
               1025, 2021, 2089, 2036, 2043, 2027, 2060, 2087, 2055, 1018, 3022, 2039, 2045, 2084, 1019, 2037}
stop_tokens = torch.tensor(list(stop_tokens), device=DEVICE)
model_id = "sentence-transformers/all-MiniLM-L6-v2"
sep = " "
msmarco_path = "./datasets/msmarco/corpus.jsonl"
doc_id_to_faiss_ids_range_file_path = "./doc_id_to_faiss_ids_range.pickle"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)


def get_model():
    model = AutoModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def parse_id_and_doc(line: str):
    doc = json.loads(line)
    doc_id, doc = doc["_id"], (doc["title"] + sep + doc["text"]).strip()
    return doc_id, doc


def tokenize_and_filter_stop_tokens(doc: str):
    tokenized_doc = tokenizer(doc, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    token_ids, attention_mask = tokenized_doc['input_ids'], tokenized_doc['attention_mask']
    keep_mask = ~torch.isin(token_ids, stop_tokens)
    token_ids, attention_mask = token_ids[keep_mask], attention_mask[keep_mask]
    return token_ids, attention_mask


def builds_embs():
    # Setup
    model = get_model()
    corpus, queries, qrels = load_dataset(dataset="msmarco")
    corpus = {doc_id: (doc["title"] + sep + doc["text"]).strip() for doc_id, doc in corpus.items()}
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")

    # Train PQ
    for_train = []
    count = 0
    for docs_batch in chunked(corpus.values(), batch_size=100):
        tokenized_docs_batch = tokenizer(docs_batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        token_ids_batch, attention_mask = tokenized_docs_batch['input_ids'], tokenized_docs_batch['attention_mask']
        with torch.no_grad():
            model_output = model(input_ids=token_ids_batch, attention_mask=attention_mask, return_dict=True)
        embs = torch.flatten(model_output.last_hidden_state, start_dim=0, end_dim=1).detach().cpu().numpy()
        for_train.append(embs)
        count += len(embs)
        if count > 5_000_000:
            break

    for_train = np.concatenate(for_train)
    index = faiss.IndexPQ(for_train.shape[1], 16, 8)
    index.train(for_train)
    del for_train
    del corpus
    del queries
    del qrels

    # Index
    doc_id_to_faiss_ids_range = {}
    with open(msmarco_path, "r") as f:
        for line in tqdm.tqdm(iterable=f, desc="indexing"):
            if not line.strip():
                continue
            doc_id, doc = parse_id_and_doc(line)
            token_ids, attention_mask = tokenize_and_filter_stop_tokens(doc)
            token_ids, attention_mask = token_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0)
            if token_ids.numel() == 0:
                continue
            with torch.no_grad():
                model_output = model(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
            context_embs = model_output.last_hidden_state.cpu()
            start = index.ntotal
            index.add(context_embs.squeeze(dim=0))
            doc_id_to_faiss_ids_range[doc_id] = (start, index.ntotal)

    faiss.write_index(index, "./ms_marco_embs_pq.index")
    with open(doc_id_to_faiss_ids_range_file_path, "wb") as f:
        pickle.dump(doc_id_to_faiss_ids_range, f)


def build_token_to_embs():
    with open(doc_id_to_faiss_ids_range_file_path, 'rb') as f:
        doc_id_to_faiss_ids_range = pickle.load(f)
    token_to_faiss_ids = defaultdict(set)
    with open(msmarco_path, "r") as f:
        for line in tqdm.tqdm(iterable=f, desc="indexing"):
            if not line.strip():
                continue
            doc_id, doc = parse_id_and_doc(line)
            token_ids, _ = tokenize_and_filter_stop_tokens(doc)
            if token_ids.numel() == 0:
                continue
            faiss_ids_range = doc_id_to_faiss_ids_range[doc_id]
            faiss_ids = list(range(faiss_ids_range[0], faiss_ids_range[1]))
            assert token_ids.numel() == len(faiss_ids)
            for token, faiss_id in zip(token_ids, faiss_ids):
                token_to_faiss_ids[token.item()].add(faiss_id)

    with open("./token_to_faiss_ids.pickle", 'wb') as f:
        pickle.dump(token_to_faiss_ids, f)


def build_lmdb_for_token_to_embs():
    with open("./token_to_faiss_ids.pickle", 'rb') as f:
        token_to_faiss_ids = pickle.load(f)
    db_path = "./token_to_faiss_ids.db"
    env = lmdb.open(path=db_path, map_size=50 * 1024 ** 3)
    with env.begin(write=True) as tx:
        for token, faiss_ids in tqdm.tqdm(token_to_faiss_ids.items(), desc="saving"):
            tx.put(str(token).encode("utf-8"), pickle.dumps(faiss_ids))
        tx.commit()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--build', type=str,
                        help='What do you want to build? (variants: embs, token2embs, lmdb_for_token2embs)')
    args = parser.parse_args()
    print(f"Params: {args}")
    if args.build == "embs":
        builds_embs()
    elif args.build == "token2embs":
        build_token_to_embs()
    elif args.build == "lmdb_for_token2embs":
        build_lmdb_for_token_to_embs()
    else:
        print("Please choose --build arg")

# TODO: filter stopwords
# list(map(lambda id: tokenizer.convert_ids_to_tokens(id), sorted(list(map(lambda k: k, token_to_faiss_ids)), key=lambda k: len(token_to_faiss_ids[k]),reverse=True)))
# list(map(lambda p: (tokenizer.convert_ids_to_tokens(p[0]), p[1]), sorted(list(map(lambda k: (k, len(token_to_faiss_ids[k])), token_to_faiss_ids)), key=lambda p: p[1],reverse=True)))
