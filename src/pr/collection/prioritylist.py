from heapq import heappop


class PriorityList:
    def __init__(self, maxsize):
        self.queue = []
        self.maxsize = maxsize

    def put(self, item):
        size = len(self.queue)
        for i in range(0, size, 1):
            if self.queue[i] > item:
                self.queue[i], item = item, self.queue[i]

        if size < self.maxsize:
            self.queue.append(item)

    def get(self):
        return heappop(self.queue)

    def empty(self):
        return not len(self.queue)
