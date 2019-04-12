#!/usr/bin/env python
"""Convert PyTorch AWD-Language Model Embeddings to word vectors."""

import argparse
import os
import torch

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("AWD-LSTM Embeddings to Word Vectors")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dictionary = torch.load(args.dictionary)
    model = torch.load(args.model, map_location='cpu')
    embeddings = model[0].encoder.weight.data.cpu().numpy()

    kv = KeyedVectors(embeddings.shape[1])
    kv.syn0 = embeddings
    kv.vocab = {
        w: Vocab(index=i) for i, w in enumerate(dictionary.dictionary.idx2word)
    }
    kv.index2word = dictionary.dictionary.idx2word

    kv.save(args.output)


if __name__ == "__main__":
    main()
