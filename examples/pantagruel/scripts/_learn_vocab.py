import logging
import os
import sentencepiece as sp
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from typing import List, Optional

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

def gen_vocab(
    input_path: Path, output_path_prefix: Path, model_type="bpe",
    vocab_size=50000, special_symbols: Optional[List[str]] = None
):
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
        # f"--byte_fallback=true",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}

    vocab = {
        i: s
        for i, s in vocab.items()
    }
    with open(output_path_prefix.as_posix() + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--model-prefix", default=None, type=str)
    parser.add_argument("--model-type", default="bpe", type=str)
    parser.add_argument("--vocab-size", default=50000, type=int)
    parser.add_argument("--special-symbols", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    assert input_path.is_file()
    logging.info(f"Learning vocabulary for text file: {args.input_path}")
    logging.info(f"Sentencepiece parameters: \
                 - model type: {args.model_type} \
                 - vocab size: {args.vocab_size} \
                 - special symbols: {args.special_symbols}")
    
    output_dir = args.output_dir
    if not output_dir:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving SPM outputs to folder: {output_dir}")

    model_prefix = args.model_prefix
    if not model_prefix:
        model_prefix = f"spm_{args.model_type}{int(args.vocab_size) // 1000}K" if int(args.vocab_size) > 1000 else f"spm_{args.model_type}{int(args.vocab_size)}"
    logging.info(f"model_prefix: {model_prefix}")

    gen_vocab(
        input_path,
        output_dir / model_prefix,
        model_type=args.model_type,
        vocab_size=int(args.vocab_size),
        special_symbols=args.special_symbols,
    )

if __name__ == "__main__":
    main()