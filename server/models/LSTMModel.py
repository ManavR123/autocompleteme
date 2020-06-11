import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
import math
from utils import ids
from tqdm import tqdm


class LSTMNetwork(nn.Module):
    # a PyTorch Module that holds the neural network for your model

    def __init__(self, vocab_size, num_hidden_lstm_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(128, 512, num_hidden_lstm_layers, dropout=0.5).cuda()
        self.linear = nn.Linear(512, 128).cuda()
        self.linear_2 = nn.Linear(128, vocab_size).cuda()
        self.dp = nn.Dropout(0.5)

    def forward(self, x, state):
        """Compute the output of the network.
    
        x - a tensor of int64 inputs with shape (seq_len, batch)
        state - a tuple of two tensors with shape (num_layers, batch, hidden_size)
                representing the hidden state and cell state of the of the LSTM.
        returns a tuple with two elements:
          - a tensor of log probabilities with shape (seq_len, batch, vocab_size)
          - a state tuple returned by applying the LSTM.
        """
        x = self.embedded_dropout(self.linear_2, x, dropout=0.1 if self.training else 0)
        x, state = self.lstm(x, state)
        x = self.dp(x)
        x = self.linear(x)
        x = self.linear_2(x)
        return x, state

    def embedded_dropout(self, embed, words, dropout=0.1):
        if dropout:
            mask = embed.weight.data.new().resize_(
                (embed.weight.size(0), 1)
            ).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight

        X = torch.nn.functional.embedding(words, masked_embed_weight,)
        return X


class LSTMModel:
    "A class that wraps LSTMNetwork to handle training and evaluation."

    def __init__(
        self, train_dataset, validation_dataset, vocab, num_hidden_lstm_layers=3
    ):
        self.network = LSTMNetwork(len(vocab), num_hidden_lstm_layers).cuda()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.vocab = vocab
        self.num_hidden_lstm_layers = num_hidden_lstm_layers

    def train(self):
        train_iterator = torchtext.data.BPTTIterator(
            self.train_dataset, batch_size=64, bptt_len=32, device="cuda"
        )
        h = Variable(
            torch.zeros(self.num_hidden_lstm_layers, 64, self.network.lstm.hidden_size),
            requires_grad=False,
        ).cuda()
        c = Variable(
            torch.zeros(self.num_hidden_lstm_layers, 64, self.network.lstm.hidden_size),
            requires_grad=False,
        ).cuda()
        state = (h, c)
        optim = torch.optim.Adam(self.network.parameters())
        prev_validation = float("inf")
        for epoch in range(20):
            print("Epoch", epoch + 1)
            self.network.train()
            for batch in tqdm.tqdm_notebook(train_iterator, leave=False):
                assert (
                    self.network.training
                ), "make sure your network is in train mode with `.train()`"
                text, target = batch.text, batch.target
                text, target = (
                    text.to(torch.int64).cuda(),
                    target.to(torch.int64).cuda(),
                )
                optim.zero_grad()

                output, state = self.network(text, state)
                output = output.view(-1, output.shape[-1])
                target = target.view(-1,)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optim.step()

                state = (state[0].detach(), state[1].detach())

            validation_pp = self.dataset_perplexity(self.validation_dataset)
            print("Validation score:", validation_pp)

            if validation_pp < prev_validation:
                torch.save(self.network.state_dict(), "lstm_language_model.pkl")
                prev_validation = validation_pp

        self.network.load_state_dict(torch.load("lstm_language_model.pkl"))

    def next_word_probabilities(self, text_prefix):
        "Return a list of probabilities for each word in the vocabulary."

        prefix_token_tensor = torch.tensor(ids(text_prefix), device="cuda").view(-1, 1)

        prefix_token_tensor = prefix_token_tensor.to(torch.int64).cuda()
        h = Variable(
            next(self.network.parameters()).data.new(
                self.num_hidden_lstm_layers, 1, self.network.lstm.hidden_size
            ),
            requires_grad=False,
        )
        c = Variable(
            next(self.network.parameters()).data.new(
                self.num_hidden_lstm_layers, 1, self.network.lstm.hidden_size
            ),
            requires_grad=False,
        )
        state = (h, c)
        with torch.no_grad():
            self.network.eval()
            output = self.network(prefix_token_tensor, state)
            output = output.squeeze()
            probs = F.softmax(output, dim=-1)
            return probs[-1]

    def dataset_perplexity(self, torchtext_dataset):

        iterator = torchtext.data.BPTTIterator(
            torchtext_dataset, batch_size=64, bptt_len=32, device="cuda"
        )

        h = Variable(
            next(self.network.parameters()).data.new(
                self.num_hidden_lstm_layers, 64, self.network.lstm.hidden_size
            ),
            requires_grad=False,
        ).cuda()
        c = Variable(
            next(self.network.parameters()).data.new(
                self.num_hidden_lstm_layers, 64, self.network.lstm.hidden_size
            ),
            requires_grad=False,
        ).cuda()
        state = (h, c)

        nll = 0.0
        num_data = 0

        with torch.no_grad():
            self.network.eval()
            for batch in tqdm(iterator, leave=False):
                sample, target = batch.text, batch.target
                output, state = self.network(sample, state)
                output = output.view(-1, output.shape[-1])
                target = target.view(-1,)
                num_data += target.shape[0]
                nll += F.cross_entropy(output, target, reduction="sum")

        nll = nll / num_data
        score = torch.exp(nll)
        score = score.data.cpu().numpy()
        return score
