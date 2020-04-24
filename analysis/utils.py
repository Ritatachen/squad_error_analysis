import json
import re
import  string
devset_path = "../../data/SQuAD2/dev-v2.0.json"
trainset_path = "../../data/SQuAD2/train-v2.0.json"

pred_path = "./predictions/bert-large/predictions_0.json"

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

with open(trainset_path) as f:
    train = json.load(f)
    train_data = train['data']

with open(devset_path) as f:
    dev = json.load(f)
    dev_data = dev['data']

with open(pred_path) as f:
    pred = json.load(f)

has_answer_results = []
no_answer_results = []
for data_blob in dev_data:
    title = data_blob['title']
    paragraphs = data_blob['paragraphs']
    for paragraph in paragraphs:
        for qa in paragraph['qas']:
            pred_answer = normalize_answer(pred[qa['id']])
            correct = False
            if qa['is_impossible']:
                if pred_answer=='':
                    correct = True
                    pred_answer = '-'
                no_answer_results.append('{}\t{}\t{}\t{}\t{}'.format(qa['id'], correct, pred_answer, qa['question'], paragraph['context']))
            else:
                all_answers = []

                for answer_dict in qa['answers']:
                    answer = normalize_answer(answer_dict['text'])
                    all_answers.append(answer)
                    correct = correct or answer == pred_answer
                if not pred_answer:
                    pred_answer = '-'
                all_answers = '['+'|'.join(all_answers)+']'
                # has_answer_results.append('{}\t{}\t{}\t{}\t{}\t{}'.format(qa['id'], correct, pred_answer, all_answers, qa['question'],paragraph['context']))
                has_answer_results.append('{}\n{}\n{}\n{}\n{}\n{}\n'.format(paragraph['context'], qa['id'], correct, pred_answer, all_answers, qa['question']))

has_answer_results = sorted(has_answer_results, key=lambda x:x.split('\n')[2])[:2000]
no_answer_results = sorted(no_answer_results, key=lambda x:x.split('\t')[1])
with open("bert-question-type-has-answer-stats.txt", 'w') as f:
    f.write('id\tT\F\tpred_answer\tgt_answer\tquestion\n')
    f.write('\n'.join(has_answer_results))

with open("bert-question-type-no-answer-stats.txt", 'w') as f:
    f.write('id\tT\F\tanswer\tquestion\n')
    f.write('\n'.join(no_answer_results))
