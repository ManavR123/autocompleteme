import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import ids
from tqdm import tqdm


class NeuralNgramDataset(torch.utils.data.Dataset):
    def __init__(self, text_token_ids, vocab, n):
        self.text_token_ids = text_token_ids
        self.n = n
        self.vocab = vocab

    def __len__(self):
        return len(self.text_token_ids)

    def __getitem__(self, i):
        if i < self.n - 1:
            prev_token_ids = [self.vocab.stoi["<eos>"]] * (
                self.n - i - 1
            ) + self.text_token_ids[:i]
        else:
            prev_token_ids = self.text_token_ids[i - self.n + 1 : i]

        assert len(prev_token_ids) == self.n - 1

        x = torch.tensor(prev_token_ids)
        y = torch.tensor(self.text_token_ids[i])
        return x, y


class NeuralNGramNetwork(nn.Module):
    # a PyTorch Module that holds the neural network for your model

    def __init__(self, n, vocab_size):
        super().__init__()
        self.n = n

        self.embed = nn.Embedding(vocab_size, 128).cuda()
        self.linear_1 = nn.Linear((self.n - 1) * 128, 1024).cuda()
        self.linear_2 = nn.Linear(1024, 1024).cuda()
        self.linear_3 = nn.Linear(1024, 128).cuda()
        self.dropout = nn.Dropout(p=0.1).cuda()
        self.linear_4 = nn.Linear(128, vocab_size).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def forward(self, x):
        # x is a tensor of inputs with shape (batch, n-1)
        # this function returns a tensor of log probabilities with shape (batch, vocab_size)

        x = self.embed(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        return x


class NeuralNGramModel:
    # a class that wraps NeuralNGramNetwork to handle training and evaluation
    # it's ok if this doesn't work for unigram modeling
    def __init__(self, n, train_text, validation_text, vocab):
        self.n = n
        self.network = NeuralNGramNetwork(n, len(vocab))
        self.vocab = vocab
        self.train_text = train_text
        self.validation_text = validation_text

    def train(self):
        dataset = NeuralNgramDataset(ids(self.train_text), self.vocab, self.n)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        # iterating over train_loader with a for loop will return a 2-tuple of batched tensors
        # the first tensor will be previous token ids with size (batch, n-1),
        # and the second will be the current token id with size (batch, )

        self.network.cuda()
        optim = torch.optim.Adam(self.network.parameters())
        prev_validation = float("inf")
        for epoch in range(10):
            print("Epoch", epoch)
            self.network.train()
            for batch in tqdm.notebook.tqdm(train_loader, leave=False):
                assert (
                    self.network.training
                ), "make sure your network is in train mode with `.train()`"
                prev, curr = batch
                prev, curr = prev.cuda(), curr.cuda()
                optim.zero_grad()
                output = self.network(prev)
                loss = F.cross_entropy(output, curr)
                loss.backward()
                optim.step()

            validation_pp = self.perplexity(self.validation_text)
            print("Validation score:", validation_pp)

            if validation_pp < prev_validation:
                torch.save(self.network.state_dict(), "neural_language_model.pkl")
                prev_validation = validation_pp

    def next_word_probabilities(self, text_prefix):
        while len(text_prefix) < self.n - 1:
            text_prefix = ["<eos>"] + text_prefix
        if len(text_prefix) > self.n - 1:
            text_prefix = text_prefix[len(text_prefix) - self.n + 1 :]
        self.network.eval()
        x = torch.Tensor(ids(text_prefix)).to(torch.int64).cuda()
        x = x.reshape((1, len(x)))
        probs = self.network.softmax(self.network(x))[0]
        return probs

    def perplexity(self, text):
        with torch.no_grad():
            self.network.eval()
            data = NeuralNgramDataset(ids(text), self.vocab, self.n)
            log_probabilities = []
            count = 0
            for text in tqdm(data):
                if count > 100:
                    break
                count += 1
                x = text[0].cuda()
                x = x.reshape((1, len(x)))
                probs = self.network.softmax(self.network(x))[0]
                log_probabilities.append(math.log(probs[text[1].item()]))
            return math.exp(-np.mean(log_probabilities))
