# encoding:utf-8


import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm

event2num = {}


def convert_file(file_path):
    with open(file_path) as f:
        data = json.load(f)

    def helper(trigger, word):
        pos = word.find(trigger)
        if pos != 0:
            return False
        else:
            for i in range(len(trigger), len(word)):
                if word[i] not in "-,.)(":
                    return False
            return True

    result = {}
    for event_name, events in tqdm(data.items()):
        result[event_name] = []
        for raw_sent, trigger, pos in events:
            if len(trigger.split()) > 1:
                continue
            sent = raw_sent.split()
            validate_pos = []
            for i, word in enumerate(sent):
                if helper(trigger, word):
                    validate_pos.append(i)
            final_pos = -1
            for i, p in enumerate(validate_pos):
                if abs(p - pos[0]) <= 3:
                    final_pos = i
            if final_pos != -1:
                cur_instance = {
                    'sent': raw_sent,
                    'trigger': trigger,
                    'pos': final_pos  # 指的是匹配上的第几个！！不是绝对位置！！
                }
                result[event_name].append(cur_instance)

    tokenize_result = []
    cnt = 0
    # all_exist_sample = set()
    for event_name, events in tqdm(result.items()):
        for event in events:
            sent = event['sent']
            trigger = event['trigger']
            pos = event['pos']
            try:
                tokenized_sent = list(word_tokenize(sent))
            except:
                continue
            cur = 0
            final_pos = -1
            for i, word in enumerate(tokenized_sent):
                if word == trigger:
                    if cur == pos:
                        final_pos = i
                    else:
                        cur += 1
            if final_pos != -1:
                cur_instance = {
                    'token': tokenized_sent,
                    'events': [
                        {
                            'type': event_name,
                            'trigger': trigger,
                            'offset': [final_pos, final_pos+1]
                        }
                    ]
                }
            
                tokenize_result.append(cur_instance)
                event2num[event_name] = event2num.get(event_name, 0) + 1
            else:
                cnt += 1
    return tokenize_result


if __name__ == '__main__':
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    sent_data = convert_file('../fewevent_origin_data/Few-Shot_ED.json')
    type2lemma_num = {}
    for sent in sent_data:
        events = sent['events']
        for event in events:
            type_ = event['type']
            trigger = event['trigger']
            lemma = lemmatizer.lemmatize(trigger, pos='v')
            if type_ not in type2lemma_num:
                type2lemma_num[type_] = set()
            type2lemma_num[type_].add(lemma)
    for key, value in type2lemma_num.items():
        type2lemma_num[key] = len(value)
    json.dump(fp=open('./type2lemma_num.json', 'w'), obj=type2lemma_num)
    json.dump(fp=open('./type2num.json', 'w'), obj=event2num)
    f = open('./sent_data.jsonl', 'w')
    for data in sent_data:
        f.write(json.dumps(data))
        f.write('\n')
    f.close()