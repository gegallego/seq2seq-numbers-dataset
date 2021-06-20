import torch
from torch.utils.data import Dataset

class Seq2SeqNumbersDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_dict, tgt_dict):
        super(Seq2SeqNumbersDataset, self).__init__()
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def __getitem__(self, idx: int):
        src = self.src_sents[idx]
        if isinstance(src, str):
            src_ids = torch.LongTensor(self.src_dict.encode(src))
            src_length = len(src_ids)
        else:
            src_ids = [torch.LongTensor(self.src_dict.encode(s)) for s in src]
            src_length = [len(s) for s in src_ids]

        tgt = self.tgt_sents[idx]
        if isinstance(tgt, str):
            tgt_ids = torch.LongTensor(self.tgt_dict.encode(tgt))
            tgt_length = len(tgt_ids)
        else:
            tgt_ids = [torch.LongTensor(self.tgt_dict.encode(s)) for s in tgt]
            tgt_length = [len(s) for s in tgt_ids]

        return {
            'src': src,
            'src_ids': src_ids,
            'src_length': src_length,
            'tgt': tgt,
            'tgt_ids': tgt_ids,
            'tgt_length': tgt_length,
            }
    
    def __len__(self):
        return len(self.src_sents)
