import argparse

def get_args() -> argparse.Namespace:
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-mid', '--backbone-model-id', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='backbone model id (default sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('-d', '--dataset', type=str, default='scifact', help='BEIR dataset name (default scifact)')
    parser.add_argument('-l', '--dataset-length', type=int, default=None, help='Dataset length (default None, all dataset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size (default 128)')
    parser.add_argument('-kmn', '--kmeans-n-clusters', type=int, default=8, help='kmeans clusters number (default 8)')
    parser.add_argument('-M', '--hnsw-M', type=int, default=32, help='the number of neighbors used in the graph. A larger M is more accurate but uses more memory (default 32)')
    parser.add_argument('-efs', '--hnsw-ef-search', type=int, default=16, help='HNSW ef search param (default 16)')
    parser.add_argument('-efc', '--hnsw-ef-construction', type=int, default=40, help='HNSW ef construction param (default 40)')
    parser.add_argument('-in', '--index-n-neighbors', type=int, default=8, help='index neighbors number (default 8)')
    parser.add_argument('-stk', '--search-top-k', type=int, default=1000, help='search tok k results (default 1000)')
    parser.add_argument('-sn', '--search-n-neighbors', type=int, default=3, help='search neighbors number (default 3)')
    parser.add_argument('--train-hnsw-only', action="store_true", help='train hnsw only (default False)')
    parser.add_argument('-imi', '--in-memory-index', action="store_true", help='in-memory inverted index type (default False)')
    parser.add_argument('-c', '--use-cache', action="store_true", help='use cache (default False)')
    parser.add_argument('--lazy-loading', action="store_true", help='use lazy loading for dataset to save memory (default False)')
    parser.add_argument('-p', '--base-path', type=str, default='./', help='base path (default ./)')
    args = parser.parse_args()
    print(f"Params: {args}")
    return args
 
