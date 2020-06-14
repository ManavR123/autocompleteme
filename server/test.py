import os
import pickle

from models.utils import (
    check_validity,
    generate_text,
    save_truncated_distribution,
)
from models.UnigramModel import UnigramModel
from models.NGramModel import NGramModel
from models.DiscountBackoffModel import DiscountBackoffModel
from models.KneserNeyBaseModel import KneserNeyBaseModel
from models.NeuralNGramModel import NeuralNGramModel
from models.LSTMModel import LSTMModel

import torchtext

text_field = torchtext.data.Field()
datasets = torchtext.datasets.WikiText2.splits(root=".", text_field=text_field)
train_dataset, validation_dataset, test_dataset = datasets
text_field.build_vocab(train_dataset)
vocab = text_field.vocab

train_text = train_dataset.examples[0].text  # a list of tokens (strings)
validation_text = validation_dataset.examples[0].text

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "saved_models/")

unigram_model = NGramModel(train_text, vocab, n=1)
check_validity(unigram_model, validation_text)
print("unigram validation perplexity:", unigram_model.perplexity(validation_text))
with open(filename + "unigram.pkl", "wb") as f:
    pickle.dump(unigram_model, f)
print("SAVED: UNIGRAM MODEL")

bigram_model = NGramModel(train_text, vocab, n=2)
check_validity(bigram_model, validation_text)
print("bigram validation perplexity:", bigram_model.perplexity(validation_text))
with open(filename + "bigram.pkl", "wb") as f:
    pickle.dump(bigram_model, f)
print("SAVED: BIGRAM MODEL")


trigram_model = NGramModel(train_text, vocab, n=3)
check_validity(trigram_model, validation_text)
print("trigram validation perplexity:", trigram_model.perplexity(validation_text))
with open(filename + "trigram.pkl", "wb") as f:
    pickle.dump(trigram_model, f)
print("SAVED: TRIGRAM MODEL")


bigram_backoff_model = DiscountBackoffModel(train_text, vocab, unigram_model, 2)
check_validity(bigram_backoff_model, validation_text)

trigram_backoff_model = DiscountBackoffModel(train_text, vocab, bigram_backoff_model, 3)
check_validity(trigram_backoff_model, validation_text)
print(
    "trigram backoff validation perplexity:",
    trigram_backoff_model.perplexity(validation_text),
)
with open(filename + "trigram_backoff.pkl", "wb") as f:
    pickle.dump(trigram_backoff_model, f)
print("SAVED: TRIGRAM_BACKOFF MODEL")


kn_base = KneserNeyBaseModel(train_text, vocab)
check_validity(kn_base, validation_text)

bigram_kn_backoff_model = DiscountBackoffModel(train_text, vocab, kn_base, 2)

trigram_kn_backoff_model = DiscountBackoffModel(
    train_text, vocab, bigram_kn_backoff_model, 3
)
print(
    "trigram Kneser-Ney backoff validation perplexity:",
    trigram_kn_backoff_model.perplexity(validation_text),
)
with open(filename + "trigram_kn_backoff.pkl", "wb") as f:
    pickle.dump(trigram_kn_backoff_model, f)
print("SAVED: TRIGRAM_KN_BACKOFF MODEL")

neural_trigram_model = NeuralNGramModel(3, train_text, validation_text, vocab)
check_validity(neural_trigram_model, validation_text)
neural_trigram_model.train()
print(
    "neural trigram validation perplexity:",
    neural_trigram_model.perplexity(validation_text),
)
with open(filename+"neural_trigram.pkl", "wb") as f:
    pickle.dump(neural_trigram_model, f)
print("SAVED: NEURAL_TRIGRAM MODEL")

LSTM = LSTMModel(train_dataset, validation_dataset, vocab)
LSTM.train()
print("lstm validation perplexity:", LSTM.dataset_perplexity(validation_dataset))
with open(filename+"LSTM.pkl", "wb") as f:
    pickle.dump(LSTM, f)
print("SAVED: LSTM MODEL")
