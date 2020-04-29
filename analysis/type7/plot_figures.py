#!/usr/bin/python
import pickle
import numpy as np
import glob
import seaborn
import matplotlib.pyplot as plt
import os
from settings import MODEL_CLASSES

_, _, tokenizer_class = MODEL_CLASSES['bert']
tokenizer = tokenizer_class.from_pretrained('bert-large-uncased', do_lower_case=True, cache_dir=None,)
path = '../../type7'

list_7 = glob.glob(os.path.join(path, "*.pkl"))
t = 0

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    candidates = [i for i, e in enumerate(l) if e==sl[0]]
    for ind in candidates:
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


for l in list_7:
    if not l:
        continue
    t += 1
    print(t)
    att_pkl = l
    # att_pkl = os.path.join(path,"{}.pkl".format(l))
    attentions = pickle.load(open(att_pkl,'rb'))
    tokens = attentions['tokens']
    q_idx_st, q_idx_end = 0, tokens.index('[SEP]')
    c_idx_st, c_idx_end = q_idx_end+1, len(tokens)-1

    q_tokens = tokens[q_idx_st:q_idx_end]
    c_tokens = tokens[c_idx_st:c_idx_end]
    c_string = tokenizer.convert_tokens_to_string(c_tokens)
    c_string = '\''.join(c_string.split(' \' '))

    answer_tokens = [answer['text'] for answer in attentions['answers']]

    answer_ids = [tokenizer.encode(text)[1:-1] for text in answer_tokens]
    c_ids = tokenizer.convert_tokens_to_ids(c_tokens)

    data = np.vstack([attentions['attention'][-1][att_idx][0, 1:].T for att_idx in range(16)])
    data = data[:, :len(tokens)+1]

    max_entry = np.max(data)
    gt = np.zeros((1, data.shape[1]))
    indices = [find_sub_list(answer_id, c_ids) for answer_id in answer_ids]

    for indexxx in indices:
        for indexx in indexxx:
            for tkidx in range(indexx[0], indexx[-1]+1):
                    gt[:, q_idx_end+tkidx] = 0.5

    data = np.vstack([data,gt])
    plt.figure(figsize = (50,50))
    ax = seaborn.heatmap(data, xticklabels= tokens[1:], yticklabels= list(range(16)), square=True, vmin=0.0, vmax=0.5,cbar=False)
    ax.figure.savefig('./figures/'+l.split('/')[-1].replace('pkl','png'))
