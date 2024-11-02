import argparse
import glob
import os
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers, Regex
from transformers import PreTrainedTokenizerFast

# datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

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
SPECIAL_TOKENS += [f"<extra_id_{i}>" for i in range(50)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--bpe-size",
        default=50000,
        type=int,
        help="Number of merge operations"
    )
    parser.add_argument(
        "--out-dir",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument("--bpe-name", default="bpe-bytelevel", type=str, help="The name of the output vocab files")
    # parser.add_argument("--unicode_normalizer", default="nfc", help="unicodedata normalization", type=str)
    parser.add_argument("--add_prefix_space", action="store_true")
    args = parser.parse_args()

    files = glob.glob(f"{args.data_dir}/*.txt")
    if not files:
        print(f"File does not exist: {files}")
        exit(1)

    print(f'*** add_prefix_space: {args.add_prefix_space}')
    # print(f'*** unicode_normalizer: {args.unicode_normalizer}')

    # Instantiate tokenizer
    tokenizer = Tokenizer(models.BPE(
        fuse_unk=True, byte_fallback=True)
    )
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace(Regex("\n{1,}"), " \n "),
            normalizers.Replace(Regex("\t{1,}"), " \t "),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(prepend_scheme="always" if args.add_prefix_space else "never"),  
        # pre_tokenizers.Split(pattern=Regex(_SPLIT_REGEX), behavior="isolated", invert=False),
        ])
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(),
                                        decoders.Metaspace(prepend_scheme="always" if args.add_prefix_space else "never"),
                                        decoders.Fuse(),
                                        decoders.Strip(content=" ", left=1, right=0)])

    # And then train
    bpe_trainer = trainers.BpeTrainer(vocab_size=int(args.bpe_size),
                                        min_frequency=2,
                                        show_progress=True,
                                        special_tokens=SPECIAL_TOKENS,
                                        limit_alphabet=10000,
                                        initial_alphabet=[],
                                        )
    
    tokenizer.train(files, trainer=bpe_trainer)

    # Save the files
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, f"{args.bpe_name}.json"))
    
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.out_dir, f"{args.bpe_name}.json"))
    tok.save_pretrained(os.path.join(args.out_dir, "tokenizer_fast"))


if __name__ == "__main__":
    main()