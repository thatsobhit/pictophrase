def idx_to_word(tokenizer, index):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None
