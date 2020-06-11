from collections import Counter
import math
import numpy as np


class NGramModel:
    def __init__(self, train_text, vocab, n=2, alpha=3e-3):
        # get counts and perform any other setup
        self.n = n
        self.smoothing = alpha
        self.vocab = vocab

        self.total_counts = len(train_text)
        self.counts = Counter()
        self.extra = list(np.random.choice(train_text, n - 1))
        for i in range(len(train_text)):
            if i < n - 1:
                gram = " ".join(self.extra[i:] + train_text[: i + 1])
            else:
                gram = " ".join(train_text[i - n + 1 : i + 1])
            self.counts[gram] += 1

        self.counts_1 = Counter()
        for c in self.counts:
            temp = " ".join(c.split(" ")[:-1])
            self.counts_1[temp] += self.counts[c]

    def n_gram_probability(self, n_gram):
        """Return the probability of the last word in an n-gram.
        
        n_gram -- a list of string tokens
        returns the conditional probability of the last token given the rest.
        """
        assert len(n_gram) == self.n, "{0}".format(n_gram)

        gram = " ".join(n_gram)
        n_1_gram = " ".join(n_gram[:-1])
        return (self.counts[gram] + self.smoothing) / (
            self.counts_1[n_1_gram] + len(self.vocab.itos) * self.smoothing
        )

    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""

        if len(text_prefix) < self.n - 1:
            text_prefix = self.extra + text_prefix
            text_prefix = text_prefix[-self.n :]
        if len(text_prefix) > self.n - 1:
            text_prefix = text_prefix[len(text_prefix) - self.n + 1 :]
        return [
            self.n_gram_probability(text_prefix + [word]) for word in self.vocab.itos
        ]

    def perplexity(self, full_text):
        """ full_text is a list of string tokens
        return perplexity as a float """

        log_probabilities = []
        gram_list = list(self.extra[:])
        for word in full_text:
            gram_list.append(word)
            log_probabilities.append(math.log(self.n_gram_probability(gram_list)))
            gram_list.pop(0)
        return math.exp(-np.mean(log_probabilities))
