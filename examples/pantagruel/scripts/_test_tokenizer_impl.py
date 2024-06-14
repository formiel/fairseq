import datasets

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from tokenizers import ByteLevelBPETokenizer

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
MASK_TOKEN, MASK_TOKEN_ID = "<mask>", 4


test_string = '🤗 is<SEP> great!'
special_tokens = ['<SEP>']

with open('test.txt', 'w') as f:
    f.write(test_string)

tok = ByteLevelBPETokenizer()
tok.train(
    files='test.txt',
    special_tokens=special_tokens,
    min_frequency=2
)
first_encode = tok.encode(test_string)
tok = ByteLevelBPETokenizer(*tok.save('.', 'test2'))
second_encode = tok.encode(test_string)
print(first_encode.tokens, second_encode.tokens, sep='\n')