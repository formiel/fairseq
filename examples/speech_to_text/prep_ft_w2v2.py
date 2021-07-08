"""
This script prepares data for supervised/self-supervised fine-tuning with wav2vec 2.0
- Input: .tsv files which are outputs of the prep_${dataset}_data.py script
- Output: .tsv, .ltr, .wrd, and .txt (to learn dictitionary) files which are 
inputs for fine-tuning with wav2vec 2.0     
"""

import argparse
import os
from examples.speech_to_text.data_utils import load_df_from_tsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path", type=str, 
                                help="Path to input tsv file.")
    parser.add_argument("--dest", type=str, 
                                help="Path to directory where outputs are saved.")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    translations = {}

    split = os.path.basename(args.tsv_path).replace(".tsv", "").split("_")[0]
    df = load_df_from_tsv(args.tsv_path)

    ids_arr = [f"{'_'.join(n.split('_')[:-1])}_{str(n.split('_')[-1]).zfill(4)}.wav" for n in df["id"].values]
    nframes_arr = df["n_frames"].values
    assert len(ids_arr) == len(nframes_arr)

    # tsv file
    with open(os.path.join(args.dest, f"{split}.tsv"), "w") as f:
        print(os.path.join(os.path.dirname(args.tsv_path), "data", split, "wav_split"), file=f)
        for i, id in enumerate(ids_arr):
            print(
                f"{id}\t{nframes_arr[i]}", file=f
            )

    # ltr and wrd file
    with open(args.tsv_path, "r") as f_in, \
        open(os.path.join(args.dest, f"{split}.ltr"), "w") as ltr_out, \
        open(os.path.join(args.dest, f"{split}.wrd"), "w") as wrd_out:
        header = next(f_in).strip()
        for line in f_in:
            line = line.strip().split('\t')
            id = line[0]
            if id not in translations:
                texts = line[6].upper() # src_text
                translations[id] = texts
            print(translations[id], file=wrd_out)
            print(
                " ".join(list(translations[id].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()