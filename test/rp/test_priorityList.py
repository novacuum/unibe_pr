from unittest import TestCase

from pr.collection.prioritylist import PriorityList


class TestPriorityList(TestCase):
    def test_put(self):
        pl = PriorityList(4)
        pl.put(5)
        pl.put(8)
        pl.put(4)
        pl.put(7)
        pl.put(1)
        self.assertEqual([1, 4, 5, 7], pl.queue, 'sorting failed')

    def test_get(self):
        pl = PriorityList(5)
        pl.put(5)
        pl.put(4)
        pl.put(7)
        pl.put(1)
        self.assertEqual(1, pl.get(), 'first should be lowest')
