# This is based on the preprocessing script for the CommonVoice dataset in SpeechBrain


import argparse
import glob
import os
import string
from pathlib import Path
import pandas as pd
import csv
import unicodedata, six, re
from argparse import Namespace

import soundfile
from fairseq.data import Dictionary, encoders


"""
Read provided ${SPLIT}.tsv files provided by CommonVoice
and return manifest and label files ready for training data2vec models 
where SPLIT=["train", "dev", "test"]
"""
def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def normalize_unicode(text):
    """
    Normalize unicode underlying representation
    """
    text = unicodedata.normalize("NFC", text)

    return text

def text_preprocesser_fr(words):
    words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()
    words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        )
    words = words.replace("’", "'")
    words = words.replace("é", "é")
    words = words.replace("æ", "ae")
    words = words.replace("œ", "oe")
    words = words.replace("â", "â")
    words = words.replace("ç", "ç")
    words = words.replace("è", "è")
    words = words.replace("à", "à")
    words = words.replace("û", "û")
    words = words.replace("î", "î")
    words = words.upper()

    # Case of apostrophe collés
    words = words.replace("L'", "L' ")
    words = words.replace("L'  ", "L' ")
    words = words.replace("S'", "S' ")
    words = words.replace("S'  ", "S' ")
    words = words.replace("D'", "D' ")
    words = words.replace("D'  ", "D' ")
    words = words.replace("J'", "J' ")
    words = words.replace("J'  ", "J' ")
    words = words.replace("N'", "N' ")
    words = words.replace("N'  ", "N' ")
    words = words.replace("C'", "C' ")
    words = words.replace("C'  ", "C' ")
    words = words.replace("QU'", "QU' ")
    words = words.replace("QU'  ", "QU' ")
    words = words.replace("M'", "M' ")
    words = words.replace("M'  ", "M' ")

    # Case of apostrophe qui encadre quelques mots
    words = words.replace(" '", " ")
    words = words.replace("A'", "A")
    words = words.replace("B'", "B")
    words = words.replace("E'", "E")
    words = words.replace("F'", "F")
    words = words.replace("G'", "G")
    words = words.replace("K'", "K")
    words = words.replace("Q'", "Q")
    words = words.replace("V'", "V")
    words = words.replace("W'", "W")
    words = words.replace("Z'", "Z")
    words = words.replace("O'", "O")
    words = words.replace("X'", "X")
    words = words.replace("AUJOURD' HUI", "AUJOURD'HUI")
    return words

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing wav files to index"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="subset to process"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--bpe-model", type=str, help="path to dictionary"
    )
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = Path(args.root) 
    tsv_path = dir_path / f"{args.split}.tsv"
    audio_path = dir_path / "clips_wav"

    tsv_data = pd.read_csv(
        tsv_path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )
    file_paths = tsv_data["path"].tolist()
    transcriptions = tsv_data["sentence"].tolist()

    print(f'Preparing manifest files containing root directory on top and relative paths in each line...')
    os.makedirs(args.dest, exist_ok=True)
    with open(os.path.join(args.dest, f"{args.split}.tsv"), "w") as dest:
        print(dir_path.as_posix(), file=dest)

        for p in file_paths:
            fp = audio_path / p.replace(".mp3", ".wav")
            frames = soundfile.info(fp).frames
            print(
                "{}\t{}".format(os.path.relpath(fp.as_posix(), dir_path), frames), file=dest
            )

    print(f"Preparing label files...")
    # with open(
    #     os.path.join(args.dest, f"{args.split}.ltr"), "w"
    # ) as ltr_out, open(
    #     os.path.join(args.dest, f"{args.split}.wrd"), "w"
    # ) as wrd_out, open(
    with open(os.path.join(args.dest, f"{args.split}.clean"), "w"
    ) as clean_out, open(
        os.path.join(args.dest, f"{args.split}.fr.ltr"), "w"
    ) as label_out:
        for line in transcriptions:
            # Remove spaces at the beginning and the end of the sentence
            line = line.lstrip().rstrip()
            # Remove trailing quotation marks
            line = str(line.strip('\"'))
            if line[-1] not in [".", "?", "!"]:
                line = line + "."
            line = convert_to_unicode(line)
            line = normalize_unicode(line)
            line = text_preprocesser_fr(line)
            # Remove multiple spaces
            line = re.sub(" +", " ", line)
            # Getting chars
            # chars = line.replace(" ", "_")
            # chars = " ".join([char for char in chars][:])

            print(line, file=clean_out)
            print(
                " ".join(list(line.replace(" ", "|"))),
                file=label_out
            )

    # print(f"Encoding label files...")
    # bpe_tokenizer = {"bpe": "sentencepiece", "sentencepiece_model": args.bpe_model}
    # bpe = encoders.build_bpe(Namespace(**bpe_tokenizer))
    # with open(os.path.join(args.dest, f"{args.split}.bpe500"), "w"
    # ) as out:
    #     for line in transcriptions:
    #         line = line.strip('\"')
    #         if line[-1] not in [".", "?", "!"]:
    #             line = line + "."
    #         # encoding with SPM
    #         line = bpe.encode(line)
    #         print(line, file=out)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
