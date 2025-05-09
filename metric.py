import json

def metric(pred, gold):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    predictions = []
    for sent_idx in pred:
        gold_num += len(gold[sent_idx])
        pred_correct_num = 0
        prediction = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index, ele.tail_start_index, ele.tail_end_index) for ele in pred[sent_idx]]))
        pred_num += len(prediction)

        for ele in prediction:
            if ele in gold[sent_idx]:
                predictions.append(ele)
                right_num += 1
                pred_correct_num += 1

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num, " relation_right_num = ", rel_num, " entity_right_num = ", ent_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    return {"precision": precision, "recall": recall, "f1": f_measure}

def t_test(pred, gold):
    assert pred.keys() == gold.keys()
    pred_correct_num_dict = {}
    for sent_idx in pred:
        
        pred_correct_num_dict[sent_idx] = {}
        
        prediction = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index, ele.tail_start_index, ele.tail_end_index) for ele in pred[sent_idx]]))
        right_num = 0
        
        for ele in prediction:
            if ele in gold[sent_idx]:
                right_num += 1
        
        pred_correct_num_dict[sent_idx]['gold_num'] = len(gold[sent_idx])
        pred_correct_num_dict[sent_idx]['pred_num'] = len(prediction)
        pred_correct_num_dict[sent_idx]['right_num'] = right_num
       
    file_path = './slot_output_for_t_test.json' 
    with open(file_path, 'w') as json_file:
        json.dump(pred_correct_num_dict, json_file, indent=4)
        
    return pred_correct_num_dict

def num_metric(pred, gold):
    test_1, test_2, test_3, test_4, test_other = [], [], [], [], []
    for sent_idx in gold:
        if len(gold[sent_idx]) == 1:
            test_1.append(sent_idx)
        elif len(gold[sent_idx]) == 2:
            test_2.append(sent_idx)
        elif len(gold[sent_idx]) == 3:
            test_3.append(sent_idx)
        elif len(gold[sent_idx]) == 4:
            test_4.append(sent_idx)
        else:
            test_other.append(sent_idx)

    pred_1 = get_key_val(pred, test_1)
    gold_1 = get_key_val(gold, test_1)
    pred_2 = get_key_val(pred, test_2)
    gold_2 = get_key_val(gold, test_2)
    pred_3 = get_key_val(pred, test_3)
    gold_3 = get_key_val(gold, test_3)
    pred_4 = get_key_val(pred, test_4)
    gold_4 = get_key_val(gold, test_4)
    pred_other = get_key_val(pred, test_other)
    gold_other = get_key_val(gold, test_other)
    num_metric_scores = []
    print("--*--*--Num of Gold Triplet is 1--*--*--")
    num_metric_scores.append(metric(pred_1, gold_1))
    print("--*--*--Num of Gold Triplet is 2--*--*--")
    num_metric_scores.append(metric(pred_2, gold_2))
    print("--*--*--Num of Gold Triplet is 3--*--*--")
    num_metric_scores.append(metric(pred_3, gold_3))
    print("--*--*--Num of Gold Triplet is 4--*--*--")
    num_metric_scores.append(metric(pred_4, gold_4))
    print("--*--*--Num of Gold Triplet is greater than or equal to 5--*--*--")
    num_metric_scores.append(metric(pred_other, gold_other))
    
    return num_metric_scores
    


def overlap_metric(pred, gold):
    normal_idx, multi_label_idx, overlap_idx = [], [], []
    for sent_idx in gold:
        triplets = gold[sent_idx]
        if is_normal_triplet(triplets):
            normal_idx.append(sent_idx)
        if is_multi_label(triplets):
            multi_label_idx.append(sent_idx)
        if is_overlapping(triplets):
            overlap_idx.append(sent_idx)
    pred_normal = get_key_val(pred, normal_idx)
    gold_normal = get_key_val(gold, normal_idx)
    pred_multilabel = get_key_val(pred, multi_label_idx)
    gold_multilabel = get_key_val(gold, multi_label_idx)
    pred_overlap = get_key_val(pred, overlap_idx)
    gold_overlap = get_key_val(gold, overlap_idx)
    overlap_metric_scores = []
    print("--*--*--Normal Triplets--*--*--")
    overlap_metric_scores.append(metric(pred_normal, gold_normal))
    print("--*--*--Multiply label Triplets--*--*--")
    overlap_metric_scores.append(metric(pred_multilabel, gold_multilabel))
    print("--*--*--Overlapping Triplets--*--*--")
    overlap_metric_scores.append(metric(pred_overlap, gold_overlap))

    return overlap_metric_scores


def is_normal_triplet(triplets):
    entities = set()
    for triplet in triplets:
        head_entity = (triplet[1], triplet[2])
        tail_entity = (triplet[3], triplet[4])
        entities.add(head_entity)
        entities.add(tail_entity)
    return len(entities) == 2 * len(triplets)


def is_multi_label(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    return len(entity_pair) != len(set(entity_pair))


def is_overlapping(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.append((pair[0], pair[1]))
        entities.append((pair[2], pair[3]))
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


def get_key_val(dict_1, list_1):
    dict_2 = dict()
    for ele in list_1:
        dict_2.update({ele: dict_1[ele]})
    return dict_2
