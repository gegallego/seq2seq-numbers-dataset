from typing import Union, List
from collections import OrderedDict

from . import Tokenizer

class Dictionary:
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"

    def __init__(self, tokenizer: Tokenizer = None):
        self.vocabulary = {
        self.BOS_TOKEN: 99999999,
        self.EOS_TOKEN: 99999999,
        self.UNK_TOKEN: 99999999,
        self.PAD_TOKEN: 99999999,
        }
        self.__compute_dicts()
        self.tokenizer = tokenizer

    def __compute_dicts(self):
        self.dictionary = {
            item[0]: n for n, item in enumerate(self.vocabulary.items())
        }
        self.dictionary_inv = {v: k for k,v in self.dictionary.items()}

    def add_sents(self, sents: Union[List[str], List[List[str]]]):
        for sent in sents:
            if isinstance(sent, str):
                sent = self.tokenizer.tokenize(sent)
            for token in sent:
                if token in self.vocabulary:
                    self.vocabulary[token] += 1
                else:
                    self.vocabulary[token] = 1

        self.vocabulary = OrderedDict(
            sorted(self.vocabulary.items(), key=lambda t: t[1], reverse=True)
        )
        self.__compute_dicts()

    def encode(self, sent: Union[List[str], str]) -> List[int]:
        if isinstance(sent, str):
            sent = self.tokenizer.tokenize(sent)
        sent_idxs = [self.dictionary[tok] if tok in self.dictionary.keys()
                    else self.dictionary[self.UNK_TOKEN] for tok in sent]
        return [self.dictionary[self.BOS_TOKEN]] + \
               sent_idxs + \
               [self.dictionary[self.EOS_TOKEN]]

    def decode(self, idxs: List[int], detokenize: bool = True) -> List[str]:
        sent = [self.dictionary_inv[idx] for idx in idxs
                if (idx != self.dictionary[self.BOS_TOKEN] and \
                    idx != self.dictionary[self.EOS_TOKEN] and \
                    idx != self.dictionary[self.PAD_TOKEN]) \
                ]
        if detokenize:
            return self.tokenizer.detokenize(sent)
        else:
            return sent

    def bos_idx(self):
        return self.dictionary[self.BOS_TOKEN]

    def eos_idx(self):
        return self.dictionary[self.EOS_TOKEN]

    def unk_idx(self):
        return self.dictionary[self.UNK_TOKEN]

    def pad_idx(self):
        return self.dictionary[self.PAD_TOKEN]

    def __getitem__(self, idx: int):
        return self.dictionary_inv[idx]

    def __len__(self):
        return len(self.dictionary)
