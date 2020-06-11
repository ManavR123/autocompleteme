from collections import defaultdict
from NGramModel import NGramModel


class DiscountBackoffModel(NGramModel):
    def __init__(self, train_text, vocab, lower_order_model, n=2, delta=0.9):
        super().__init__(train_text, vocab, n=n)
        self.lower_order_model = lower_order_model
        self.discount = delta

        self.n_1 = defaultdict(lambda: set())
        for c in self.counts:
            temp = c.split(" ")
            self.n_1[" ".join(temp[:-1])].add(temp[-1])

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n

        # back off to the lower_order model with n'=n-1 using its n_gram_probability function
        gram = " ".join(n_gram)
        n_1_gram = " ".join(n_gram[:-1])

        if self.counts_1[n_1_gram] == 0:
            return self.lower_order_model.n_gram_probability(n_gram[1:])

        prob = max(self.counts[gram] - self.discount, 0) / (self.counts_1[n_1_gram])
        alpha = self.discount * len(self.n_1[n_1_gram]) / self.counts_1[n_1_gram]
        backoff_prob = alpha * self.lower_order_model.n_gram_probability(n_gram[1:])
        return prob + backoff_prob
