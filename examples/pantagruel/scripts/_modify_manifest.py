import argparse
import csv
from pathlib import Path

"""
Input: a list of manifest tsv files
Output: one tsv file with one commot root path
- Prepend one level up to the path (from the first line 
of the manifest file) to each line in the manifest
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-root", type=str, required=True)
    parser.add_argument("--tsv-files", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="train-full")
    args = parser.parse_args()

    tsv_root = Path(args.tsv_root)
    tsv_paths = [Path(tsv_root / f"{file}.tsv") for file in args.tsv_files.split(",")]
    for p in tsv_paths:
        assert p.exists()
    parent_paths = [None] * len(tsv_paths)
    manifest_data = []
    num_examples = 0

    for n, p in enumerate(tsv_paths):
        with open(p, "r") as fhandle:
            freader = csv.reader(fhandle, delimiter="\t", quotechar='"')
            for i, row in enumerate(freader):
                if i == 0:
                    assert len(row) == 1
                    root_path = Path(row[0])
                    parent_paths[n] = root_path.parent.absolute()
                else:
                    assert len(row) == 2
                    manifest_data.append(
                        [f"{root_path.name}/{row[0]}", row[1]]
                    )
                    num_examples += 1
    assert all(e is not None for e in parent_paths)
    print(f"Total number of examples: {num_examples}")

    # Get the common roots
    common_root = None
    for p in parent_paths:
        if not common_root:
            common_root = p
        else:
            if p != common_root:
                raise ValueError("Path to parent directories need to be the same!")
    common_root = "/gpfsscratch/rech/ahm/umz16dj/Data/LibriSpeech_raw"
    # common_root = "/gpfsssd/scratch/rech/ahm/umz16dj/Data/LibriSpeech_raw/librispeech_finetuning"
    print(f"common root: {common_root}")

    # Write to one common tsv
    output_path = tsv_root / f"{args.output_file}.tsv"
    with open(output_path, "w", newline='') as file:
        file.write(f"{common_root}\n")
        for data in manifest_data:
            file.write('\t'.join(data) + '\n')
    print(f"Wrote all data to {output_path}.")

if __name__ == "__main__":
    main()