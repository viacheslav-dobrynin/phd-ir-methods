from unittest import TestCase

import torch

from inverted_index import InvertedIndexOnDict


class TestInvertedIndexOnDict(TestCase):

    def setUp(self):
        self.embs = torch.tensor([[0, 1, 0], [1, 0, 1]])

    def test_write_and_search(self):
        index = InvertedIndexOnDict(lambda q: torch.tensor([0, 1, 0] if q == "first" else torch.tensor([1, 0, 1])))
        index.write(self.embs)

        self.assertEqual(len(index.index), 3)
        self.assertEqual(index.index[0], [(1, 1.0)])
        self.assertEqual(index.index[1], [(0, 1.0)])
        self.assertEqual(index.index[2], [(1, 1.0)])

        result = index.search({"1": "first"}, top_k=1)
        self.assertEqual(result["1"][0], 1.0)
        self.assertNotIn(1, result)

        result = index.search({"2": "second"}, top_k=1)
        self.assertEqual(result["2"][1], 2.0)
        self.assertNotIn(0, result)

    def test_write_error_on_rewrite(self):
        index = InvertedIndexOnDict(lambda q: torch.tensor([]))
        index.write(self.embs)

        with self.assertRaises(RuntimeError):
            index.write(self.embs)

    def test_empty_search(self):
        index = InvertedIndexOnDict(lambda _: torch.tensor([0, 0, 0]))
        index.write(self.embs)

        result = index.search({"1": "query"}, top_k=1)
        self.assertEqual(len(result["1"]), 0)
