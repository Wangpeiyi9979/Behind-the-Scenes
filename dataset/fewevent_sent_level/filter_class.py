#encoding:utf-8

import json
def get_filter_class(path, threshold):
    type2num = json.load(open(path, 'r'))
    return_class = list()
    for type, num in type2num.items():
        if num >= threshold:
#             return_class.append(type)
            return_class.append((type, num))
#     return_class.sort() # 唯一返回顺序
    return_class.sort(key=lambda x: x[-1], reverse=True)
    return_class = [x[0] for x in return_class]
    return return_class

def get_filter_trigger_num_class(classes, path, threshold):
    type2lemma_num = json.load(open(path, 'r'))
    return_class = list()
    for cls in classes:
        if cls in type2lemma_num and type2lemma_num[cls] >= threshold:
            return_class.append(cls)
    return return_class

def read_data(data_path):
    all_data = []
    f = open(data_path, encoding='utf-8')
    datas = f.readlines()
    for data in datas:
        all_data.append(json.loads(data))
    return all_data

def save_data(all_data, data_path):
    f = open(data_path, 'w')
    for data in all_data:
        f.write(json.dumps(data))
        f.write('\n')

def filter_data(all_data, keep_classes):
    keep_datas = []
    for data in all_data:
        new_events = []
        for event in data['events']:
            if event['type'] in keep_classes:
                new_events.append(event.copy())
        if len(new_events) != 0:
            copy_data = data.copy()
            copy_data['events'] = new_events
            keep_datas.append(copy_data)
    return keep_datas


def save_class(classes, case):
    with open('../../tool_data/fewevent/{}_classes.txt'.format(case), 'w') as f:
        for class_ in classes:
            f.write(class_ + '\n')

if __name__ == '__main__':
    all_data = read_data('./sent_data.jsonl')
    print(len(all_data))
    all_keep_classes = list(get_filter_class('./type2num.json', 40))
    keep_classes = list(get_filter_class('./type2num.json', 40))
    print(len(keep_classes)) # 134
    all_keep_classes = list(get_filter_trigger_num_class(all_keep_classes, './type2lemma_num.json', 2))
    keep_classes = list(get_filter_trigger_num_class(keep_classes, './type2lemma_num.json', 5))
    all_class = [x.split('.') for x in keep_classes]
    main2num = {}
    for main_class, _ in all_class:
        main2num[main_class] = main2num.get(main_class, 0) + 1
    print(main2num)
    test_classes = set()
    val_classes = set()
    test_main_classes = {'Education', 'Business', 'Justice', 'Olympics'}
    val_main_classes = {'Movement', 'Organization', 'Military', 'Transaction'}
    for main_class, class_ in all_class:
        if main_class in test_main_classes:
            test_classes.add(main_class + '.' + class_)
        elif main_class in val_main_classes:
            val_classes.add(main_class + '.' + class_)
    print(test_classes)
    print(val_classes)
    train_classes = (set(all_keep_classes) - val_classes) - test_classes
    val_classes = val_classes & set(keep_classes)
    test_classes = test_classes & set(keep_classes)
    save_class(train_classes, 'train')
    save_class(val_classes, 'val')
    save_class(test_classes, 'test')

    train_datas = filter_data(all_data, train_classes)
    val_datas = filter_data(all_data, val_classes)
    test_datas = filter_data(all_data, test_classes)
    print('total data:{}; split total data:{}'.format(len(filter_data(all_data, keep_classes)), (len(train_datas) + len(val_datas) + len(test_datas))))
    print('train class:{}; train data num:{}'.format(len(train_classes), len(train_datas)))
    print('val class:{}; val data num:{}'.format(len(val_classes), len(val_datas)))
    print('test class:{}; test data num:{}'.format(len(test_classes), len(test_datas)))

    save_data(train_datas, './few_shot_train_data.jsonl')
    save_data(test_datas, './few_shot_test_data.jsonl')
    save_data(val_datas, './few_shot_val_data.jsonl')
