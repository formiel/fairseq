import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("bookcorpus")
    df = ds.data["train"]["text"].to_pandas()
    print(f"First five rows...")
    print(df.head(5)) # pandas series

    with open(args.output_path, "w") as f:
        for row in df:
            f.write(f"{row}\n")


if __name__ == "__main__":
    main()