#!/usr/bin/python
import pickle
import numpy as np
import glob
import seaborn
import matplotlib.pyplot as plt
import os

path = '../../selected_correct_has_answer'
# list_7 = [line.strip().split('\t')[0].strip() for line in open("./7.txt").readlines()]
# print(lst)
list_7 = glob.glob(os.path.join(path, "*.pkl"))
t = 0
for l in list_7:
    if not l:
        continue
    t += 1
    print(t)
    att_pkl = l
    # att_pkl = os.path.join(path,"{}.pkl".format(l))
    attentions = pickle.load(open(att_pkl,'rb'))
    tokens = attentions['tokens']
    q_idx_st, q_idx_end = 1, tokens.index('[SEP]')
    c_idx_st, c_idx_end = q_idx_end+1, len(tokens)-1

    q_tokens = [elem for _ in range(16) for elem in tokens[q_idx_st:q_idx_end]]+['']
    c_tokens = tokens[c_idx_st:c_idx_end]

    answer_tokens = set(' '.join([answer['text']for answer in attentions['answers']]).split())
    data = np.vstack([attentions['attention'][4][att_idx][c_idx_st:c_idx_end,q_idx_st:q_idx_end].T for att_idx in range(16)])

    max_entry = np.max(data)
    gt = np.zeros((1,data.shape[1]))
    for idx, token in enumerate(c_tokens):
        if token in answer_tokens:
            gt[:, idx] = max_entry+0.1
    data = np.vstack([data,gt])
    # data = np.sum(data, axis=0, keepdims=True)
    plt.figure(figsize = (50,50))
    ax = seaborn.heatmap(data, xticklabels= c_tokens, yticklabels= q_tokens, square=True, vmin=0.0, vmax=max_entry+0.1,cbar=False)
    #ax = seaborn.heatmap(data[:sep2,:sep2].T, xticklabels= tokens, yticklabels= tokens,square=True, vmin=0.0, vmax=0.1,cbar=False)
    ax.figure.savefig('./has_answer/'+l.split('/')[-1].replace('pkl','png'))
