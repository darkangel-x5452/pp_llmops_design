# import nltk
# nltk.download('all')
from nltk.tokenize import word_tokenize

class ExampleSimpleTokenizer():
    def __init__(self, text: str):
        self.data = text

    def tokenize_text(self, data: str) -> list:
        tokens = word_tokenize(data)
        words = [word for word in tokens if word.isalpha()]
        print(f"number of tokens: {len(words)}")
        print(f"number of unique tokens: {len(set(words))}")
        print(f"starting tokens: {words[:20]}")
        return words

    def create_vocab_ids(self, words: list) -> dict[str, int]:
        unique_words = sorted(set(words))
        print(f"unique vocab size: {len(unique_words)}")

        vocab_ids = {_unique: _id for _id, _unique in enumerate(unique_words)}

        return vocab_ids

    def encode_text_to_ids(self, text: list, vocab_ids: dict[str, int]):
        encoded_ls = [vocab_ids[_text] for _text in text]
        print(f"encoded starting list: {encoded_ls[:10]}")
