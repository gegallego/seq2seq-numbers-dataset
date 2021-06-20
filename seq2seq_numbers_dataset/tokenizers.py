from typing import List, Union

class Tokenizer:
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def detokenize(cls, tokens: List[str]) -> str:
        raise NotImplementedError()

class NumTokenizer(Tokenizer):
    @classmethod
    def tokenize(cls, text: Union[str, float]) -> List[str]:
        if isinstance(text, float):
            text = "{:,.2f}".format(text)
        return [c for c in text]

    @classmethod
    def detokenize(cls, tokens: List[str]) -> str:
        return ''.join(tokens)


class WordTokenizer(Tokenizer):
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return text.replace(',', ' , ').replace('-', ' - ').split()

    @classmethod
    def detokenize(cls, tokens: List[str]) -> str:
        return ' '.join(tokens).replace(' , ', ', ').replace(' - ', '-')
