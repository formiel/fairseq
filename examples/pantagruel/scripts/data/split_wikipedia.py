# Author: Phuong-Hang Le (hangtp.le@gmail.com)
# Date: 26 April 2024


import argparse
import random
import re
from pathlib import Path


def split_into_docs(root, lang, version, val_percentage=0.05):
    wikiname = f"{lang}wiki"
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    lines = []
    count = 0
    with open(root / f"{wikiname}_{version}" / f"{lang}wiki.full.raw") as f:
        for line in f:
            lines.append(line.strip())
            count += 1

    val_flag = False
    train_data = ( root / f"{wikiname}_{version}"/ f'{wikiname}.train').open('w')
    val_data = ( root / f"{wikiname}_{version}"/ f'{wikiname}.dev').open('w')

    # Calculate number of articles
    num_articles = 0
    for _,l in enumerate(lines):
        if l.startswith('<doc id="'):
            num_articles += 1

    # Choose random articles for validation purpose
    val_numbers = int(num_articles * val_percentage/100)
    val_articles = sorted(random.sample(range(0, num_articles), val_numbers))

    num_articles = 0
    for _, l in enumerate(lines):
        if l.startswith('<doc id="'):
            if num_articles in val_articles:
                title = title_re.findall(l)[0].replace('/','_')
                print(f"| validation no. {num_articles}: {title}")
                val_flag = True
                val_data.write(f"{l}\n")
            else:
                val_flag = False
                train_data.write(f"{l}\n")
            num_articles += 1
        else:
            if val_flag:
                val_data.write(f"{l}\n")
            else:
                train_data.write(f"{l}\n")

    val_data.close()
    train_data.close()
    print(f"*** Total number of articles: {num_articles} ***")
    print(f'*** Number of validation articles: {val_numbers} ***')
    print(f"| List of validation articles: {val_articles}")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--lang", default="en", type=str)
    parser.add_argument("--version", required=True, type=str)
    args = parser.parse_args()

    split_into_docs(Path(args.root), lang=args.lang, version=args.version)

if __name__ == "__main__":
    main()
