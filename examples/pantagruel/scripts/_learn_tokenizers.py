import argparse
import glob
import os
from tokenizers import ByteLevelBPETokenizer

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
MASK_TOKEN, MASK_TOKEN_ID = "<mask>", 4
SPECIAL_TOKENS = [
            BOS_TOKEN,
            PAD_TOKEN,
            EOS_TOKEN,
            UNK_TOKEN,
            MASK_TOKEN,
        ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        default=None,
        metavar="path",
        type=str,
        required=True,
        help="The files to use as training; accept '**/*.txt' type of patterns \
                            if enclosed in quotes",
    )
    parser.add_argument(
        "--bpe-size",
        default=50000,
        type=int,
        help="Number of merge operations"
    )
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument("--name", default="bpe-bytelevel", type=str, help="The name of the output vocab files")
    parser.add_argument("--test-decoding", action="store_true", help="Test decoding")
    args = parser.parse_args()
    args = parser.parse_args()

    files = glob.glob(args.files)
    if not files:
        print(f"File does not exist: {args.files}")
        exit(1)

    # Initialize an empty tokenizer
    tokenizer = ByteLevelBPETokenizer(
        add_prefix_space=False,
        unicode_normalizer="nfc"
    ) 
 
    # And then train
    tokenizer.train(
        files,
        vocab_size=int(args.bpe_size),
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    # Save the files
    os.makedirs(args.out, exist_ok=True)
    tokenizer.save_model(args.out, args.name)

    # # Restoring model from learned vocab/merges
    # tokenizer = ByteLevelBPETokenizer(
    #     "/".join([args.out, "{}-vocab.json".format(args.name)]),
    #     "/".join([args.out, "{}-merges.txt".format(args.name)]),
    #     add_prefix_space=True,
    # )
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # # Test encoding
    # if args.test_decoding:
    #     # print(tokenizer.encode("Training ByteLevel BPE is very easy").tokens)
    #     original_text = "Former en utilisant ByteLevel BPE est très facile."
    #     encoded = tokenizer.encode(original_text)
    #     ids = encoded.ids
    #     print(f"Original sentence: {original_text}")
    #     print(f"Tokens: {encoded.tokens}")
    #     print(f"Number of tokens: {len(ids)}\n{ids}")
    #     decoded_text = tokenizer.decode(ids)
    #     print(f"Decoded sentence: {decoded_text}")

    #     # Decoding
    #     encoded_ids = "8387 1256 1677 16 847 306 313 22021 328 17263 13 297 1002 306 313 11624 1529 8751 328 16345 472 345 321 34057 676 16 306 2937 25996 676 322 2806 10690 327 525"
    #     decoded_text = tokenizer.decode([int(i) for i in encoded_ids.split()])
    #     print(f"Decoded sentence-2: {decoded_text}")
    #     print(f"IDs: {tokenizer.encode(decoded_text).ids}")

if __name__ == "__main__":
    main()