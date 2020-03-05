import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import torch


def create_predictions_file(questions_file, facts_file, examples_file, logits_file, pred_output_file,
                            mcq_choices="correct", write_debug_file=False):
    """
    Utility to generate submission file from predictions (logits scores)
    """
    df_questions = pd.read_csv(questions_file, sep='\t')
    df_facts = pd.read_csv(facts_file, sep='\t').drop_duplicates(subset=["uid"], keep="first").reset_index()
    examples = torch.load(examples_file)
    logits = np.load(logits_file)
    logit_1 = logits[:, 1] - logits[:, 0]

    if write_debug_file:
        f_tmp = open(pred_output_file + "-as-text", "w")

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

    if mcq_choices != "all":
        df_questions["ProcessedQuestion"] = df_questions.apply(remove_wrong_answer_choices, 1,
                                                               choices=mcq_choices)
    else:
        df_questions["ProcessedQuestion"] = df_questions["Question"]
    vectorizer = TfidfVectorizer().fit(df_questions['Question']).fit(df_facts['text'])
    X_q = vectorizer.transform(df_questions['ProcessedQuestion'])
    X_e = vectorizer.transform(df_facts['text'])
    X_dist = cosine_distances(X_q, X_e)

    idx_start = 0
    predictions = []
    prev_query = examples[0].text_a
    for i, example in enumerate(examples):
        if example.text_a == prev_query:
            continue

        qid = examples[idx_start].guid.split('###')[0]
        q = df_questions.loc[df_questions["questionID"] == qid]
        assert q["ProcessedQuestion"].item() == examples[idx_start].text_a

        relevant_logits = logit_1[idx_start:i]
        relevant_examples = examples[idx_start:i]
        sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                    reverse=True))
        added_uids = set()
        example_preds = []
        for se in sorted_examples:
            for fid in se.guid.split('###')[1:]:
                if fid not in added_uids:
                    added_uids.add(fid)
                    example_preds.append('\t'.join([qid, fid]))
        for dist_idx in np.argsort(X_dist[q.index.to_numpy()[0]]):
            fid = df_facts.loc[dist_idx, "uid"]
            if fid not in added_uids:
                added_uids.add(fid)
                example_preds.append('\t'.join([qid, fid]))
        predictions.extend(example_preds)

        if write_debug_file:
            f_tmp.write(q["questionID"].item())
            f_tmp.write('\n')
            f_tmp.write(q["Question"].item())
            f_tmp.write('\n')
            f_tmp.write(q["ProcessedQuestion"].item())
            f_tmp.write("\n*************\n")
            for i_tmp in range(40):
                f_tmp.write(sorted_examples[i_tmp].guid.split('###')[1:].__str__())
                f_tmp.write(' Score:{:.3f}\n'.format(sorted_preds[i_tmp]))
                f_tmp.write(sorted_examples[i_tmp].text_b.__str__())
                f_tmp.write('\n')
            f_tmp.write("*************\n")
            for i_tmp in range(40):
                f_tmp.write(df_facts.loc[df_facts["uid"] == example_preds[i_tmp].split('\t')[1], "text"].item())
                f_tmp.write('\n')
            f_tmp.write("*************\n")
            for expl in q["explanation"].item().split(' '):
                f_tmp.write(df_facts.loc[df_facts["uid"] == expl.split('|')[0], "text"].item())
                f_tmp.write('\n')
            f_tmp.write("*************\n")

        prev_query = example.text_a
        idx_start = i

    qid = examples[idx_start].guid.split('###')[0]
    q = df_questions.loc[df_questions["questionID"] == qid]
    assert q["ProcessedQuestion"].item() == examples[idx_start].text_a

    relevant_logits = logit_1[idx_start:]
    relevant_examples = examples[idx_start:]
    sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                reverse=True))
    added_uids = set()
    example_preds = []
    for se in sorted_examples:
        for fid in se.guid.split('###')[1:]:
            if fid not in added_uids:
                added_uids.add(fid)
                example_preds.append('\t'.join([qid, fid]))
    for dist_idx in np.argsort(X_dist[q.index.to_numpy()[0]]):
        fid = df_facts.loc[dist_idx, "uid"]
        if fid not in added_uids:
            added_uids.add(fid)
            example_preds.append('\t'.join([qid, fid]))
    predictions.extend(example_preds)

    if write_debug_file:
        f_tmp.write(q["questionID"].item())
        f_tmp.write('\n')
        f_tmp.write(q["Question"].item())
        f_tmp.write('\n')
        f_tmp.write(q["ProcessedQuestion"].item())
        f_tmp.write("\n*************\n")
        for i_tmp in range(40):
            f_tmp.write(sorted_examples[i_tmp].guid.split('###')[1:].__str__())
            f_tmp.write(' Score:{:.3f}\n'.format(sorted_preds[i_tmp]))
            f_tmp.write(sorted_examples[i_tmp].text_b.__str__())
            f_tmp.write('\n')
        f_tmp.write("*************\n")
        for i_tmp in range(40):
            f_tmp.write(df_facts.loc[df_facts["uid"] == example_preds[i_tmp].split('\t')[1], "text"].item())
            f_tmp.write('\n')
        f_tmp.write("*************\n")
        for expl in q["explanation"].item().split(' '):
            f_tmp.write(df_facts.loc[df_facts["uid"] == expl.split('|')[0], "text"].item())
            f_tmp.write('\n')
        f_tmp.write("*************\n")

        f_tmp.close()

    logging.info("Writing to file")
    with open(pred_output_file, "w") as f:
        f.write('\n'.join(predictions))
        f.write('\n')
    logging.info("len(df_questions)={}".format(len(df_questions)))
    logging.info("len(predictions)={}".format(len(predictions)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, required=True,
                        help="The tsv file containing the evaluation")
    parser.add_argument("--facts_file", type=str, required=True,
                        help="The tsv file containing the common sense facts")
    parser.add_argument("--examples_file", type=str, help="Examples file that is being evaluated")
    parser.add_argument("--logits_file", type=str, help="Model predictions (liekly some file of the form *_preds.npy)")
    parser.add_argument("--pred_output_file", type=str, required=True,
                        help="Name of the file where predictions will be written")
    parser.add_argument("--mcq_choices", type=str, choices=['none', 'correct', 'all'], default="correct",
                        help="The choices to keep in the questions")
    parser.add_argument("--write_debug_file", action='store_true')
    args = parser.parse_args()

    create_predictions_file(questions_file=args.questions_file, facts_file=args.facts_file,
                            examples_file=args.examples_file, logits_file=args.logits_file,
                            pred_output_file=args.pred_output_file, mcq_choices=args.mcq_choices,
                            write_debug_file=args.write_debug_file)