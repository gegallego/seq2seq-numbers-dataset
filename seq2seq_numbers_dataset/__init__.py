from .dataset import Seq2SeqNumbersDataset
from .tokenizers import (
    Tokenizer,
    NumTokenizer,
    WordTokenizer,
)
from .dictionary import  Dictionary
from .collater import Seq2SeqNumbersCollater

import torch
import pandas as pd
from num2words import num2words
from sklearn.model_selection import train_test_split

torch.manual_seed(3)

def generate_dataset_df():
    random_numbers = (torch.rand(200000,) - 0.5) * 2
    random_numbers = torch.mul(
        random_numbers.reshape(-1, 100),
        torch.Tensor([1.3**i for i in range(100)])
        ).flatten()
    df = pd.DataFrame(random_numbers.sort().values.tolist(), columns=['nums'])
    df['nums'] = df['nums'].round(2).astype('float')

 
    df['words'] = df['nums'].apply(num2words)
    df['nums'] = df['nums'].apply(lambda x: "{:,.2f}".format(x))

    df_train, df_devtest = train_test_split(df, test_size=0.3, random_state=3)
    df_dev, df_test = train_test_split(df_devtest, test_size=0.5, random_state=3)

    return {
        "train": df_train,
        "dev": df_dev,
        "test": df_test,
    }

def generate_dataset_pytorch(numbers_to_words: bool = True):
    dataset_df = generate_dataset_df()

    if numbers_to_words:
        src_field = "nums"
        tgt_field = "words"
    else:
        src_field = "words"
        tgt_field = "nums"

    src_dict = Dictionary(NumTokenizer)
    tgt_dict = Dictionary(WordTokenizer)

    src_dict.add_sents(dataset_df["train"][src_field].tolist())
    tgt_dict.add_sents(dataset_df["train"][tgt_field].tolist())

    dataset_train = Seq2SeqNumbersDataset(
        dataset_df["train"][src_field].to_list(),
        dataset_df["train"][tgt_field].to_list(),
        src_dict, tgt_dict,
    )

    dataset_dev = Seq2SeqNumbersDataset(
        dataset_df["dev"][src_field].to_list(),
        dataset_df["dev"][tgt_field].to_list(),
        src_dict, tgt_dict,
    )

    dataset_test = Seq2SeqNumbersDataset(
        dataset_df["test"][src_field].to_list(),
        dataset_df["test"][tgt_field].to_list(),
        src_dict, tgt_dict,
    )

    return {
        "train": dataset_train,
        "dev": dataset_dev,
        "test": dataset_test,
    }
