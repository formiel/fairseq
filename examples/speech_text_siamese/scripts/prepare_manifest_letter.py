"""
This script prepares data for supervised/self-supervised fine-tuning with wav2vec 2.0
- Input: .tsv files which are outputs of the prep_${dataset}_data.py script
- Output: .tsv, .ltr, .wrd, and .txt (to learn dictitionary) files which are 
inputs for fine-tuning with wav2vec 2.0     
"""

import argparse
import os
import pandas as pd
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "src_text"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv-file", type=str, 
                                help="Path to input tsv file, which is the output of prep_dataset_data.py")
    parser.add_argument("--output-tsv-name", type=str, 
                                help="")

    args = parser.parse_args()

    dir_name = os.path.dirname(args.input_tsv_file)
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    with open(args.input_tsv_file, "r") as f_in:
        next(f_in)
        for line in f_in:
            line = line.strip().split('\t')
            for i, col in enumerate(MANIFEST_COLUMNS):
                if i != len(MANIFEST_COLUMNS) - 1:
                    manifest[col].append(line[i])
                else:
                    manifest[col].append(" ".join(list(line[3].replace(" ", "|"))) + " |")

    save_df_to_tsv(pd.DataFrame.from_dict(manifest), os.path.join(dir_name, args.output_tsv_name))

if __name__ == "__main__":
    main()