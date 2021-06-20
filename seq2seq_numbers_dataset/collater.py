from . import Dictionary

import torch
import torch.nn as nn

class Seq2SeqNumbersCollater:

    def __init__(self, src_dict: Dictionary, tgt_dict: Dictionary):
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @staticmethod
    def lengths_to_padding_mask(lens):
        # Copied from: https://github.com/pytorch/fairseq/blob/fc77ee/fairseq/data/data_utils.py#L532
        bsz, max_lens = lens.size(0), torch.max(lens).item()
        mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
        mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
        return mask

    def __call__(self, batch):
        src_ids = nn.utils.rnn.pad_sequence(
            [s['src_ids'] for s in batch],
            padding_value=self.src_dict.pad_idx()
        )
        src_lengths = torch.LongTensor([s['src_length'] for s in batch])
        src_padding_mask = self.lengths_to_padding_mask(src_lengths).T
        src = {
            "ids": src_ids,
            "padding_mask": src_padding_mask,
        }

        tgt_ids = nn.utils.rnn.pad_sequence(
            [s['tgt_ids'] for s in batch],
            padding_value=self.tgt_dict.pad_idx()
        )
        tgt_lengths = torch.LongTensor([s['tgt_length'] for s in batch])
        tgt_padding_mask = self.lengths_to_padding_mask(tgt_lengths).T
        tgt = {
            "ids": tgt_ids,
            "padding_mask": tgt_padding_mask,
        }

        return src, tgt
