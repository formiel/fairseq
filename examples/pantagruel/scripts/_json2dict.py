import argparse
from pathlib import Path
import ast, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="path to json file")
    args = parser.parse_args()

    dirname = Path(args.json).parent
    print(f"Parent directory: {dirname}")
    with open(args.json, "r") as f:
        data = f.readline()

    d = ast.literal_eval(data)
    print(f"type: {type(d)}")
    print(f"len(data): {len(d)}")

    with open(os.path.join(dirname, "dict.txt"), "w", encoding="utf-8") as fd:
        for _, v in d.items():
            print("{} {}".format(v, 1), file=fd)


if __name__ == "__main__":
    main()