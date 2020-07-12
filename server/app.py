import json
import math
import os
import pickle
import torch

from flask import Flask, jsonify, request, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torchtext
from models.DiscountBackoffModel import DiscountBackoffModel
from models.KneserNeyBaseModel import KneserNeyBaseModel
from models.LSTMModel import LSTMModel
from models.NeuralNGramModel import NeuralNGramModel, NeuralNGramNetwork, NeuralNgramDataset
from models.NGramModel import NGramModel
from models.UnigramModel import UnigramModel
from models.utils import ids
from torchtext.data import get_tokenizer

app = Flask(__name__, static_url_path="", static_folder="static")


tokenizer = get_tokenizer("basic_english")

text_field = torchtext.data.Field()
datasets = torchtext.datasets.WikiText2.splits(root=".", text_field=text_field)
train_dataset, validation_dataset, test_dataset = datasets
text_field.build_vocab(train_dataset)
vocab = text_field.vocab

train_text = train_dataset.examples[0].text  # a list of tokens (strings)
validation_text = validation_dataset.examples[0].text

dirname = os.path.dirname(__file__)

perplexities = json.load(open(os.path.join(dirname, "perplexity.json")))


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/next_word", methods=["POST"])
def next_word():
    text = request.json.get("text", "")
    model_name = request.json.get("model_name", None)

    if not model_name:
        return jsonify({"response": "Error! Must select a model"})
    elif model_name == "None":
        return jsonify("")

    if model_name == "GPT2":
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2", pad_token_id=gpt_tokenizer.eos_token_id)
        input_ids = gpt_tokenizer.encode(text[-100:], return_tensors='pt')
        sample_output = model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        output = gpt_tokenizer.decode(
            sample_output[0], skip_special_tokens=True)
        return jsonify(output[len(text):])

    filename = os.path.join(dirname, "saved_models/" + model_name + ".pkl")
    model = torch.load(filename, map_location=torch.device('cpu'))
    tokens = tokenizer(text)

    if model_name == "LSTM":
        next_word = ""
        a = LSTMModel(train_dataset, validation_dataset, vocab)
        a.network.load_state_dict(model)
        model = a

    elif model_name == "neural_trigram":
        a = NeuralNGramModel(3, train_text, validation_text, vocab)
        a.network.load_state_dict(model)
        model = a

    probs = torch.tensor(model.next_word_probabilities(tokens))
    idx = probs.argmax().item()
    next_word = model.vocab.itos[idx]

    return jsonify(next_word)


@app.route("/get_perplexity", methods=["GET"])
def get_perplexity():
    model_name = request.args.get("model_name", None)
    if not model_name:
        return jsonify({"response": "Error! Must select a model"})
    elif model_name not in perplexities:
        return jsonify({"response": "Error! Invalid model selected"})
    return jsonify(perplexities[model_name])


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, debug=True)
