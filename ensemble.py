import argparse
import numpy as np
import torch
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def ensemble_preds(questions_file, facts_file, overlay_examples_file, overlay_logits_file, base_examples_file,
                   base_logits_file, pred_output_file, write_debug_file=False):
    df_questions = pd.read_csv(questions_file, sep='\t')
    df_facts = pd.read_csv(facts_file, sep='\t').drop_duplicates(subset=["uid"], keep="first").reset_index()

    base_examples = torch.load(base_examples_file)
    base_logits = np.load(base_logits_file)
    base_logit_1 = base_logits[:, 1] - base_logits[:, 0]

    idx_start = 0
    prev_query = base_examples[0].text_a

    base_predictions = {}
    for i, example in enumerate(base_examples):
        if example.text_a == prev_query:
            continue

        relevant_logits = base_logit_1[idx_start:i]
        relevant_examples = base_examples[idx_start:i]
        sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                    reverse=True))
        qid = sorted_examples[0].guid.split('###')[0]
        base_predictions[qid] = ['\t'.join(se.guid.split('###')) for se in sorted_examples]

        prev_query = example.text_a
        idx_start = i

    relevant_logits = base_logit_1[idx_start:]
    relevant_examples = base_examples[idx_start:]
    sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                reverse=True))
    qid = sorted_examples[0].guid.split('###')[0]
    base_predictions[qid] = ['\t'.join(se.guid.split('###')) for se in sorted_examples]

    print("len(base_predictions): {}".format(len(base_predictions)))

    overlay_examples = torch.load(overlay_examples_file)
    overlay_logits = np.load(overlay_logits_file)
    overlay_logit_1 = overlay_logits[:, 1] - overlay_logits[:, 0]

    idx_start = 0
    overlay_predictions = []
    prev_query = overlay_examples[0].text_a
    for i, example in enumerate(overlay_examples):
        if example.text_a == prev_query:
            continue
        qid = overlay_examples[idx_start].guid.split('###')[0]
        q = df_questions.loc[df_questions["questionID"] == qid]
        assert q["ProcessedQuestion"].item() == overlay_examples[idx_start].text_a

        relevant_logits = overlay_logit_1[idx_start:i]
        relevant_examples = overlay_examples[idx_start:i]
        sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                    reverse=True))
        added_uids = set()
        example_preds = []
        for sp, se in zip(sorted_preds, sorted_examples):
            if sp < 0:
                break
            for fid in se.guid.split('###')[1:]:
                if fid not in added_uids:
                    added_uids.add(fid)
                    example_preds.append('\t'.join([qid, fid]))
        for fid_ in base_predictions[qid]:
            fid = fid_.split('\t')[1]
            if fid not in added_uids:
                added_uids.add(fid)
                example_preds.append(fid_)
        overlay_predictions.extend(example_preds)
        #
        # if write_debug_file:
        #     f_tmp.write(q["questionID"].item())
        #     f_tmp.write('\n')
        #     f_tmp.write(q["Question"].item())
        #     f_tmp.write('\n')
        #     f_tmp.write(q["ProcessedQuestion"].item())
        #     f_tmp.write("\n*************\n")
        #     for i_tmp in range(40):
        #         f_tmp.write(sorted_examples[i_tmp].guid.split('###')[1:].__str__())
        #         f_tmp.write(' Score:{:.3f}\n'.format(sorted_preds[i_tmp]))
        #         f_tmp.write(sorted_examples[i_tmp].text_b.__str__())
        #         f_tmp.write('\n')
        #     f_tmp.write("*************\n")
        #     for i_tmp in range(40):
        #         f_tmp.write(df_facts.loc[df_facts["uid"] == example_preds[i_tmp].split('\t')[1], "text"].item())
        #         f_tmp.write('\n')
        #     f_tmp.write("*************\n")
        #     for expl in q["explanation"].item().split(' '):
        #         f_tmp.write(df_facts.loc[df_facts["uid"] == expl.split('|')[0], "text"].item())
        #         f_tmp.write('\n')
        #     f_tmp.write("*************\n")

        prev_query = example.text_a
        idx_start = i

    qid = overlay_examples[idx_start].guid.split('###')[0]
    q = df_questions.loc[df_questions["questionID"] == qid]
    assert q["ProcessedQuestion"].item() == overlay_examples[idx_start].text_a

    relevant_logits = overlay_logit_1[idx_start:]
    relevant_examples = overlay_examples[idx_start:]
    sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                reverse=True))
    added_uids = set()
    example_preds = []
    for sp, se in zip(sorted_preds, sorted_examples):
        if sp < 0:
            break
        for fid in se.guid.split('###')[1:]:
            if fid not in added_uids:
                added_uids.add(fid)
                example_preds.append('\t'.join([qid, fid]))
    for fid_ in base_predictions[qid]:
        fid = fid_.split('\t')[1]
        if fid not in added_uids:
            added_uids.add(fid)
            example_preds.append(fid_)
    overlay_predictions.extend(example_preds)

    # if write_debug_file:
    #     f_tmp.write(q["questionID"].item())
    #     f_tmp.write('\n')
    #     f_tmp.write(q["Question"].item())
    #     f_tmp.write('\n')
    #     f_tmp.write(q["ProcessedQuestion"].item())
    #     f_tmp.write("\n*************\n")
    #     for i_tmp in range(40):
    #         f_tmp.write(sorted_examples[i_tmp].guid.split('###')[1:].__str__())
    #         f_tmp.write(' Score:{:.3f}\n'.format(sorted_preds[i_tmp]))
    #         f_tmp.write(sorted_examples[i_tmp].text_b.__str__())
    #         f_tmp.write('\n')
    #     f_tmp.write("*************\n")
    #     for i_tmp in range(40):
    #         f_tmp.write(df_facts.loc[df_facts["uid"] == example_preds[i_tmp].split('\t')[1], "text"].item())
    #         f_tmp.write('\n')
    #     f_tmp.write("*************\n")
    #     for expl in q["explanation"].item().split(' '):
    #         f_tmp.write(df_facts.loc[df_facts["uid"] == expl.split('|')[0], "text"].item())
    #         f_tmp.write('\n')
    #     f_tmp.write("*************\n")
    #
    #     f_tmp.close()

    print("Writing to file")
    with open(pred_output_file, "w") as f:
        f.write('\n'.join(overlay_predictions))
        f.write('\n')
    print("len(df_questions)={}".format(len(df_questions)))
    print("len(predictions)={}".format(len(overlay_predictions)))


def move_redundant_facts_to_end(questions_file, facts_file, fact_frequency_file, predictions_file,
                                output_predictions_file):
    df_questions = pd.read_csv(questions_file, sep='\t')
    df_facts = pd.read_csv(facts_file, sep='\t').drop_duplicates(subset=["uid"], keep="first").reset_index()
    fact_frequency = pickle.load(open(fact_frequency_file, "rb"))
    duplicate_facts = df_facts.loc[df_facts.duplicated(subset=["text"], keep=False)]
    for i, fact in duplicate_facts.iterrows():
        if fact_frequency.get(fact["uid"], int()) > 0:
            duplicate_facts = duplicate_facts.drop(index=[i])
    duplicate_facts = set(duplicate_facts["uid"])
    predictions = {}
    with open(predictions_file) as f:
        for line in f:
            ls = line.strip().split('\t')
            try:
                predictions[ls[0]].append(ls[1])
            except KeyError:
                predictions[ls[0]] = [ls[1]]

    new_predictions = {}
    for k, v in predictions.items():
        pred_start = []
        pred_end = []
        for fid in v:
            if fid in duplicate_facts:
                pred_end.append(fid)
            else:
                pred_start.append(fid)
        new_predictions[k] = '\n'.join([k + '\t' + p for p in pred_start + pred_end[::-1]])

    output_str = ''
    for qid in df_questions["questionID"]:
        if qid in new_predictions:
            print(qid)
            output_str += new_predictions[qid] + '\n'

    with open(output_predictions_file, "w") as f:
        f.write(output_str)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, required=True,
                        help="The tsv file containing the evaluation")
    parser.add_argument("--facts_file", type=str, required=True,
                        help="The tsv file containing the common sense facts")
    # Args to mix
    parser.add_argument("--overlay_examples_file", type=str, help="Overlay examples used as the starting point")
    parser.add_argument("--overlay_logits_file", type=str,
                        help="Overlay predictions used as the starting point. Once the predicted value drops below 0,"
                             "we swtich to the base predictions")
    parser.add_argument("--base_examples_file", type=str,
                        help="Base examples used for completing the overlay predictions")
    parser.add_argument("--base_logits_file", type=str,
                        help="Base predictions used for completing the overlay predictions. "
                             "Once overlay predictions lose confidence, we switch to the base prediction")
    # Args to move_redundant
    parser.add_argument("--fact_frequency_file", type=str)
    parser.add_argument("--predictions_file", type=str,
                        help="Name of input predictions file to be reshuffled")
    # Common args
    parser.add_argument("--pred_output_file", type=str, required=True,
                        help="Name of the file where modified predictions will be written")
    # Function choice
    parser.add_argument("--ensemble", action='store_true', help="Perform ensembling by overwtiting prediction files")
    parser.add_argument("--move_redundant", action='store_true',
                        help="Move redundant facts (these facts never occur in the training set annotation, "
                             "moreover they the same content as another fact) to the end of file to improve ranking")
    args = parser.parse_args()

    if args.ensemble and not args.move_redundant:
        ensemble_preds(args.questions_file, args.facts_file, args.overlay_examples_file, args.overlay_logits_file,
                       args.base_examples_file, args.base_logits_file, args.pred_output_file)
    elif args.move_redundant and not args.ensemble:
        move_redundant_facts_to_end(args.questions_file, args.facts_file, args.fact_frequency_file,
                                    args.predictions_file, args.output_predictions_file)
    else:
        raise ValueError("Only one of --ensemble and --move_redundant can be used at a time")
