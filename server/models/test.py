from UnigramModel import UnigramModel
from NGramModel import NGramModel
from DiscountBackoffModel import DiscountBackoffModel
from KneserNeyBaseModel import KneserNeyBaseModel
from NeuralNGramModel import NeuralNGramModel
from LSTMModel import LSTMModel

from utils import (
    check_validity,
    generate_text,
    save_truncated_distribution,
)
import torchtext

text_field = torchtext.data.Field()
datasets = torchtext.datasets.WikiText2.splits(root=".", text_field=text_field)
train_dataset, validation_dataset, test_dataset = datasets
text_field.build_vocab(train_dataset)
vocab = text_field.vocab

train_text = train_dataset.examples[0].text  # a list of tokens (strings)
validation_text = validation_dataset.examples[0].text

unigram_demonstration_model = UnigramModel(train_text, vocab)
print(
    "unigram validation perplexity:",
    unigram_demonstration_model.perplexity(validation_text),
)
check_validity(unigram_demonstration_model, validation_text)

unigram_model = NGramModel(train_text, vocab, 1)
check_validity(unigram_model, validation_text)
print("unigram validation perplexity:", unigram_model.perplexity(validation_text))
bigram_model = NGramModel(train_text, vocab, n=2)
check_validity(bigram_model, validation_text)
print("bigram validation perplexity:", bigram_model.perplexity(validation_text))

trigram_model = NGramModel(train_text, vocab, n=3)
check_validity(trigram_model, validation_text)
print("trigram validation perplexity:", trigram_model.perplexity(validation_text))

save_truncated_distribution(bigram_model, "predictions/bigram_predictions.npy")

bigram_backoff_model = DiscountBackoffModel(train_text, vocab, unigram_model, 2)
check_validity(bigram_backoff_model, validation_text)
trigram_backoff_model = DiscountBackoffModel(train_text, vocab, bigram_backoff_model, 3)
check_validity(trigram_backoff_model, validation_text)
print(
    "trigram backoff validation perplexity:",
    trigram_backoff_model.perplexity(validation_text),
)

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
save_truncated_distribution(trigram_kn_backoff_model, "predictions/trigram_kn_predictions.npy")

neural_trigram_model = NeuralNGramModel(3, train_text, validation_text, vocab)
check_validity(neural_trigram_model, validation_text)
neural_trigram_model.train()
print(
    "neural trigram validation perplexity:",
    neural_trigram_model.perplexity(validation_text),
)

save_truncated_distribution(neural_trigram_model, "predictions/neural_trigram_predictions.npy")

lstm_model = LSTMModel(train_text, vocab)
lstm_model.train()
print("lstm validation perplexity:", lstm_model.dataset_perplexity(validation_dataset))
save_truncated_distribution(lstm_model, "predictions/lstm_predictions.npy")
