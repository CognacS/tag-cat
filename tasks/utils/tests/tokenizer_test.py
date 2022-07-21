from tasks.utils import tokenizer

def main():
    print('This test starts from a source sequence, which is tokenized and indexed (in order to be used in a deep learning context).')
    print('Finally, the indices sequence in transformed back to the source symbols.')
    print()

    # STARTING SEQUENCE
    test = '1512+54=<PAD><PAD>'
    print(f'input string: "{test}"')
    # TASK ALPHABET
    alpha = [str(i) for i in range(10)] + ['+', '='] + ['<PAD>']
    # BUILD A VOCABULARY
    vocab_a2i = tokenizer.vocab_alphabet2index(alpha)
    print(f'alphabet->index vocabulary: {vocab_a2i}')

    # TOKENIZE THE STARTING SEQUENCE INTO INDICES
    index_tokens = tokenizer.index_tokenize(test,
        vocab_alpha2index=vocab_a2i,
        tokenizer_function=tokenizer.tokenizer_rich_str2chars
    )
    print(f'index tokens: "{index_tokens}"')

    # BUILD AN INVERSE VOCABULARY, GOING FROM INDICES TO SYMBOLS
    vocab_i2a = tokenizer.vocab_reverse_map(vocab_a2i)
    print(f'index->alphabet vocabulary: {vocab_i2a}')

    # DETOKENIZE THE INDICES SEQUENCE
    print(f'Here, the behavior for the <PAD> symbol (index {vocab_a2i["<PAD>"]}) is to be removed.')
    out_seq = tokenizer.index_detokenize(index_tokens,
        vocab_index2alpha=vocab_i2a,
        detokenizer_function=tokenizer.detokenizer_rich_chars2str,
        specialsymb_behavior='remove'
    )

    print(f'output string: "{out_seq}"')

if __name__ == '__main__':
    main()