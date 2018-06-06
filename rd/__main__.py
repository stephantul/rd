"""calculate the representation distance."""
import argparse
import os
import unicodedata

from .rd import rd_features

FEATURES = {"one hot",
            "constrained open ngrams",
            "weighted open bigrams",
            "open ngrams",
            "fourteen",
            "sixteen",
            "dislex"}


def normalize(string):
    """Normalize, remove accents and other stuff."""
    s = unicodedata.normalize("NFKD", string).encode('ASCII', 'ignore')
    return s.decode('utf-8')


def write_scores(path, words, scores, overwrite):
    """Write scores to a file."""
    if os.path.exists(path) and not overwrite:
        raise ValueError("File already exists.")

    with open(path, 'w') as f:
        for word, score in zip(words, scores):
            f.write("{}\t{}\n".format(word, score))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the rd20")
    parser.add_argument("-i", "--input", type=str,
                        help="The path to the input file.",
                        required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="The path to the output file.",
                        required=True)
    parser.add_argument("-f", "--features", type=str,
                        help="The features to use.",
                        choices=list(FEATURES),
                        required=True)
    parser.add_argument("-n", metavar="n", type=int, required=True)
    parser.add_argument("--normalize",
                        const=True,
                        default=False,
                        action='store_const')
    parser.add_argument("--overwrite",
                        const=True,
                        default=False,
                        action='store_const')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--loc", type=int, default=0)
    parser.add_argument("--safe",
                        const=True,
                        default=False,
                        action='store_const')
    parser.add_argument("--header",
                        const=True,
                        default=False,
                        action='store_const')
    parser.add_argument("--metric",
                        default="cosine",
                        type=str)

    args = parser.parse_args()

    words = []

    for idx, x in enumerate(open(args.input).readlines()):
        if idx == 0 and args.header:
            continue
        words.append(x.strip().split(args.sep)[args.loc])

    if args.normalize:
        words = [normalize(x) for x in words]

    rd_scores = rd_features(args.features,
                            words,
                            [args.n],
                            args.metric,
                            args.safe)
    write_scores(args.output, words, rd_scores, args.overwrite)
