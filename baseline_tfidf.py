#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def read_explanations(path):
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', type=str, required=True)
    parser.add_argument('--questions', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('-n', '--nearest', type=int, default=5000)
    parser.add_argument('--mcq-choices', type=str, choices=['none', 'correct', 'all'], default="all")
    args = parser.parse_args()

    explanations = []

    for path, _, files in os.walk(args.tables):
        for file in files:
            explanations += read_explanations(os.path.join(path, file))

    df_q = pd.read_csv(args.questions, sep='\t', dtype=str)
    df_e = pd.DataFrame(explanations, columns=('uid', 'text'))

    vectorizer = TfidfVectorizer().fit(df_q['Question']).fit(df_e['text'])

    # Remove wrong choices
    def remove_wrong_answer_choices(row, choices):
        correct_choice = row["AnswerKey"]
        option_start_loc = row["Question"].rfind("(A)")
        split0, split1 = row["Question"][:option_start_loc], row["Question"][option_start_loc:]

        if choices == "none":
            return split0

        if correct_choice == "A" and "(B)" in split1:
            split0 += (split1[3:split1.rfind("(B)")])
        elif correct_choice == "A":
            split0 += (split1[3:])
        elif correct_choice == "B" and "(C)" in split1:
            split0 += (split1[split1.rfind("(B)") + 3:split1.rfind("(C)")])
        elif correct_choice == "B":
            split0 += (split1[split1.rfind("(B)") + 3:])
        elif correct_choice == "C" and "(D)" in split1:
            split0 += (split1[split1.rfind("(C)") + 3:split1.rfind("(D)")])
        elif correct_choice == "C":
            split0 += (split1[split1.rfind("(C)") + 3:])
        elif correct_choice == "D" and "(E)" in split1:
            split0 += (split1[split1.rfind("D)") + 3:split1.rfind("(E)")])
        elif correct_choice == "D":
            split0 += (split1[split1.rfind("D)") + 3:])
        elif correct_choice == "E" and "(F)" in split1:
            split0 += (split1[split1.rfind("(E)") + 3:split1.rfind("(F)")])
        elif correct_choice == "E":
            split0 += (split1[split1.rfind("(E)") + 3:])
        else:
            raise ValueError("Unhandled option type: {}".format(correct_choice))
        return split0

    if args.mcq_choices != "all":
        df_q["ProcessedQuestion"] = df_q.apply(remove_wrong_answer_choices, 1, choices=args.mcq_choices)
    else:
        df_q["ProcessedQuestion"] = df_q["Question"]

    X_q = vectorizer.transform(df_q['ProcessedQuestion'])
    X_e = vectorizer.transform(df_e['text'])
    X_dist = cosine_distances(X_q, X_e)

    with open(args.output, "w") as f:
        for i_question, distances in enumerate(X_dist):
            for i_explanation in np.argsort(distances)[:args.nearest]:
                f.write('{}\t{}\n'.format(df_q.loc[i_question]['questionID'], df_e.loc[i_explanation]['uid']))


if '__main__' == __name__:
    main()
