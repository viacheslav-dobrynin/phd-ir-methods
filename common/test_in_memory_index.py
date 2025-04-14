from unittest import TestCase

import torch

from common.in_memory_index import InMemoryInvertedIndex


class TestInMemoryInvertedIndex(TestCase):

    def setUp(self):
        self.embs = torch.tensor([[0, 1, 0], [1, 0, 1]])

    def test_write_and_search(self):
        index = InMemoryInvertedIndex()
        for i, emb in enumerate(self.embs):
            index.add(i, emb)

        self.assertEqual(len(index.index), 3)
        self.assertEqual(index.index[0], [(1, 1.0)])
        self.assertEqual(index.index[1], [(0, 1.0)])
        self.assertEqual(index.index[2], [(1, 1.0)])

        result = index.search(torch.tensor([0, 1, 0]), top_k=1)
        self.assertEqual([(0, 1.0)], result)

        result = index.search(torch.tensor([1, 0, 1]), top_k=1)
        self.assertEqual([(1, 2.0)], result)

    def test_empty_search(self):
        index = InMemoryInvertedIndex()
        for i, emb in enumerate(self.embs):
            index.add(i, emb)

        result = index.search(torch.tensor([0, 0, 0]), top_k=1)
        self.assertEqual(len(result), 0)
