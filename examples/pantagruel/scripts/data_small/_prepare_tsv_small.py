import argparse
import csv


def get_total_duration(tsv_path, sampling_rate=16000):
    num_frames = 0
    with open(tsv_path, "r") as fhandle:
        freader = csv.reader(fhandle, delimiter="\t", quotechar='"')
        for i, row in enumerate(freader):
            if i == 0:
                continue
            else:
                num_frames += int(row[1])
    return (num_frames // sampling_rate) // 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="tsv input file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--limit", default=400, 
                        help="Stop taking examples when reaching this limit (in hours)")
    args = parser.parse_args()

    max_frames = args.limit * 3600 * 16000
    num_frames = 0
    with open(args.input, "r") as infile, open(
        args.output, "w"
    ) as outfile:
        freader = csv.reader(infile, delimiter="\t", quotechar='"')
        for i, row in enumerate(freader):
            if i == 0:
                print(row, file=outfile)
            else:
                num_frames += int(row[1])
                if num_frames < max_frames:
                    print(f"{row[0]}\t{row[1]}", file=outfile)
                else:
                    break


if __name__ == "__main__":
    main()
