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
    args = parser.parse_args()

    total_duration = get_total_duration(args.input)
    print(f'Total duration of tsv file: {total_duration} (hours)')


if __name__ == "__main__":
    main()
