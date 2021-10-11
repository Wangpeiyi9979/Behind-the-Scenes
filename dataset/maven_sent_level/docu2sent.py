#encoding:utf-8
"""
@Time: 2020/12/2 16:11
@Author: Wang Peiyi
@Site : 
@File : docu2sent.py
@Function: 将事件转为sentence level
每个instance由以下内容
{
    token: 句子的token，
    events: [
                {
                    type: 事件类型，
                    trigger：事件trigger，
                    offset：trigger span
                }
            ]
    neg_triggers:[
                    {
                          trigger: 假的trigger，
                          offset：假trigger的位置
                    }
                 ]
}
"""
import json
def convert_file(file_paths):
    all_data = []
    for file_path in file_paths:
        f = open(file_path, encoding='utf-8')
        documents = f.readlines()
        for document in documents:
            document_data = convert_document(json.loads(document))
            all_data.extend(document_data)
    return all_data
def convert_document(document):
    sent2event = {}
    sent2neg = {}
    document_data = []
    for event in document['events']:
        for mention in event['mention']:
            sent_id = mention['sent_id']
            if sent_id not in sent2event:
                sent2event[sent_id] = [{'type': event['type'], 'mention': mention}]
            else:
                sent2event[sent_id].append({'type': event['type'], 'mention': mention})
    for neg_trigger in document['negative_triggers']:
        sent_id = neg_trigger['sent_id']
        if sent_id not in sent2neg:
            sent2neg[sent_id] = [neg_trigger]
        else:
            sent2neg[sent_id].append(neg_trigger)
    for sent_id, sent in enumerate(document['content']):
        if sent_id not in sent2event:
            continue
        data_unit = {}
        events = sent2event[sent_id]
        neg_triggers = sent2neg.get(sent_id, [])
        data_unit['token'] = sent['tokens']
        data_unit['events'] = []
        for event in events:
            mention = event['mention']
            data_unit['events'].append({'type': event['type'], 'trigger': mention['trigger_word'], 'offset': mention['offset']})
        data_unit['neg_triggers'] = []
        for neg_trigger in neg_triggers:
            data_unit['neg_triggers'].append({'trigger': neg_trigger['trigger_word'], 'offset': neg_trigger['offset']})
        document_data.append(data_unit)
    return document_data

if __name__ == '__main__':
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    sent_data = convert_file(['../maven_origin_data/train.jsonl', '../maven_origin_data/valid.jsonl'])
    f = open('./sent_data.jsonl', 'w')
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
    for data in sent_data:
        f.write(json.dumps(data))
        f.write('\n')
    f.close()




