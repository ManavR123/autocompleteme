import math
import numpy as np
import random
from tqdm import tqdm


def check_validity(model, validation_text):
    """Performs several sanity checks on your model:
    1) That next_word_probabilities returns a valid distribution
    2) That perplexity matches a perplexity calculated from next_word_probabilities

    Although it is possible to calculate perplexity from next_word_probabilities, 
    it is still good to have a separate more efficient method that only computes 
    the probabilities of observed words.
    """

    log_probabilities = []
    for i in range(10):
        prefix = validation_text[:i]
        probs = model.next_word_probabilities(prefix)
        assert min(probs) >= 0, "Negative value in next_word_probabilities"
        assert max(probs) <= 1 + 1e-8, "Value larger than 1 in next_word_probabilities"
        assert abs(sum(probs) - 1) < 1e-4, "next_word_probabilities do not sum to 1"

        word_id = model.vocab.stoi[validation_text[i]]
        selected_prob = probs[word_id]
        log_probabilities.append(math.log(selected_prob))

    perplexity = math.exp(-np.mean(log_probabilities))
    your_perplexity = model.perplexity(validation_text[:10])
    assert abs(perplexity - your_perplexity) < 0.1, (
        "your perplexity does not "
        + "match the one we calculated from `next_word_probabilities`,\n"
        + "at least one of `perplexity` or `next_word_probabilities` is incorrect.\n"
        + f"we calcuated {perplexity} from `next_word_probabilities`,\n"
        + f"but your perplexity function returned {your_perplexity} (on a small sample)."
    )


def generate_text(model, n=20, prefix=("<eos>", "<eos>")):
    prefix = list(prefix)
    for _ in range(n):
        probs = model.next_word_probabilities(prefix)
        word = random.choices(model.vocab.itos, probs)[0]
        prefix.append(word)
    return " ".join(prefix)


def save_truncated_distribution(model, filename):
    """Generate a file of truncated distributions.
    
    Probability distributions over the full vocabulary are large,
    so we will truncate the distribution to a smaller vocabulary.

    Please do not edit this function
    """
    with open("eval_output_vocab.txt", "r", encoding="utf8") as eval_vocab_file:
        eval_vocab = [w.strip() for w in eval_vocab_file]
    eval_vocab_ids = [model.vocab.stoi[s] for s in eval_vocab]

    all_selected_probabilities = []
    with open("eval_prefixes.txt", "r", encoding="utf8") as eval_prefixes_file:
        lines = eval_prefixes_file.readlines()
        for line in tqdm(lines, leave=False):
            prefix = line.strip().split(" ")
            probs = model.next_word_probabilities(prefix)
            selected_probs = np.array(
                [probs[i] for i in eval_vocab_ids], dtype=np.float32
            )
            all_selected_probabilities.append(selected_probs)

    all_selected_probabilities = np.stack(all_selected_probabilities)
    np.save(filename, all_selected_probabilities)
    print("saved", filename)


def ids(vocab, tokens):
    return [vocab.stoi[t] for t in tokens]
