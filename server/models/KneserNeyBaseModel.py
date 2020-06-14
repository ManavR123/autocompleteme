from collections import defaultdict
import numpy as np
from models.NGramModel import NGramModel


class KneserNeyBaseModel(NGramModel):
    def __init__(self, train_text, vocab):
        super().__init__(train_text, vocab, n=1)

        temp = list(np.random.choice(train_text)) + train_text
        self.continuations = defaultdict(self.create_set)
        self.bigrams = set()
        for i in range(1, len(temp)):
            self.continuations[temp[i]].add(temp[i - 1])
            self.bigrams.add(" ".join(temp[i - 1 : i + 1]))

    def create_set(self):
        return set()

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 1

        return len(self.continuations[n_gram[0]]) / len(self.bigrams)
