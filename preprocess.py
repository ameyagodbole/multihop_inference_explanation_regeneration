import pandas as pd
import os
import warnings
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import namedtuple

Fact = namedtuple("Fact", "uid fact file")
answer_key_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

tables_dir = "annotation/expl-tablestore-export-2017-08-25-230344/tables/"

stopwords = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

# Lemmatization map
lemmatization = {}
with open('annotation/lemmatization-en.txt', 'r') as f:
    for line in f:
        l0 = line.strip().split('\t')
        lemmatization[l0[1]] = l0[0]
print(f"len(lemmatization): {len(lemmatization)}")

######################
# FACT AS NODE GRAPH #
######################
# Map from "words" to facts containing the "words"
graph_word_to_fact_map = {}
fact_base = {}
for path, _, files in os.walk(tables_dir):
    for f in files:
        print(".", end="")
        df = pd.read_csv(os.path.join(path, f), sep='\t')
        uid = None
        header = []
        graph_header = []

        check_skip_dep = False
        # if "[SKIP] DEP" in df.columns:
        #     check_skip_dep = True

        for name in df.columns:
            if name.startswith("[SKIP]"):
                if 'UID' in name:
                    if uid is None:
                        uid = name
                    else:
                        raise AttributeError('Possibly misformatted file: ' + path)
            elif name.startswith("[FILL]"):
                header.append(name)
            else:
                graph_header.append(name)
                header.append(name)

        if not uid or len(df) == 0:
            warnings.warn('Possibly misformatted file: ' + f)
            continue

        for _, row in df.iterrows():
            row_uid = row[uid]
            # if check_skip_dep and not pd.isna(row["[SKIP] DEP"]):
            # skip deprecated row
            # continue
            if row_uid in fact_base:
                print(f"repeated UID {row_uid} in file {f}")
                continue
            fact_base[row_uid] = Fact(row_uid, ' '.join(str(s) for s in list(row[header]) if not pd.isna(s)), f)
            for col in graph_header:
                if not pd.isna(row[col]):
                    for graph_word in tokenizer.tokenize(str(row[col]).lower()):
                        if graph_word in stopwords:
                            continue
                        try:
                            graph_word_to_fact_map[graph_word].add(row_uid)
                        except KeyError:
                            graph_word_to_fact_map[graph_word] = set([row_uid])

print(f"len(fact_base): {len(fact_base)}")
print(f"len(graph_word_to_fact_map): {len(graph_word_to_fact_map)}")

link_words = list(graph_word_to_fact_map.keys())
for link_word in link_words:
    if link_word in lemmatization:
        linked_uids = graph_word_to_fact_map.pop(link_word)
        if lemmatization[link_word] in graph_word_to_fact_map:
            graph_word_to_fact_map[lemmatization[link_word]].update(linked_uids)
        else:
            graph_word_to_fact_map[lemmatization[link_word]] = linked_uids

print("After lemmatization:")
print(f"len(fact_base): {len(fact_base)}")
print(f"len(graph_word_to_fact_map): {len(graph_word_to_fact_map)}")

words_to_prune = []
# OPTIONALLY TO GET PRUNED GRAPH WITH HIGH FREQUENCY WORD EDGES DROPPED
# words_to_prune = ["object", "animal", "hemisphere", "something", "water", "plant", "northern", "move", "increase",
#                   "require", "energy", "environment","decrease", "food", "southern", "change", "body", "state",
#                   "organism"]
adjacency_map = {}

for link_word, linked_uids in graph_word_to_fact_map.items():
    if link_word in words_to_prune:
        continue
    for linked_uid in linked_uids:
        try:
            adjacency_map[linked_uid].update(linked_uids)
        except KeyError:
            adjacency_map[linked_uid] = linked_uids.copy()
        adjacency_map[linked_uid].remove(linked_uid)
        if len(adjacency_map[linked_uid]) == 0:
            del adjacency_map[linked_uid]
print(f"len(adjacency_map): {len(adjacency_map)}")

fact_kb = []
for link_word, linked_uids in graph_word_to_fact_map.items():
    if link_word in words_to_prune:
        continue
    for l1 in linked_uids:
        for l2 in linked_uids:
            if l1 != l2:
                fact_kb.append((l1, link_word, l2))
print("len(fact_kb): {}".format(len(fact_kb)))

if not os.path.exists("fact_graph/fact_as_node"):
    os.makedirs("fact_graph/fact_as_node")
pickle.dump(graph_word_to_fact_map, open("fact_graph/fact_as_node/graph_word_to_fact_map.pkl", "wb"))
pickle.dump(adjacency_map, open("fact_graph/fact_as_node/adjacency_map.pkl", "wb"))
pickle.dump(fact_kb, open("fact_graph/fact_as_node/fact_kb.pkl", "wb"))

######################
# FACT AS EDGE GRAPH #
######################

inv_graph_word_to_fact_map = {}
for link_word, linked_uids in graph_word_to_fact_map.items():
    for linked_uid in linked_uids:
        try:
            inv_graph_word_to_fact_map[linked_uid].append(link_word)
        except KeyError:
            inv_graph_word_to_fact_map[linked_uid] = [link_word]
fact_as_edge_kb = []
for linked_uid, linked_words in inv_graph_word_to_fact_map.items():
    for l1 in linked_words:
        for l2 in linked_words:
            if l1 != l2:
                fact_as_edge_kb.append((l1, linked_uid, l2))

if not os.path.exists("fact_graph/fact_as_edge"):
    os.makedirs("fact_graph/fact_as_edge")
pickle.dump(inv_graph_word_to_fact_map, open("fact_graph/fact_as_edge/fact_to_graph_word_map.pkl", "wb"))
pickle.dump(fact_as_edge_kb, open("fact_graph/fact_as_edge/fact_kb.pkl", "wb"))