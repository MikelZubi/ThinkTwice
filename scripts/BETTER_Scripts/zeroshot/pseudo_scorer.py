def make_triplets(templates):
    triplets = []
    for template in templates:
        template_type = template['template_type']
        for key in template:
            if key != 'template_type':
                triplet = (template_type, key, template[key])
                triplets.append(triplet)
    return triplets

def evaluate_triples(prediction,gold):
    correct = 0
    for triplet in prediction:
        if triplet in gold:
            correct += 1
        else:
            for gold_triplet in gold:
                if (
                    triplet[0] == gold_triplet[0]
                    and triplet[1] == gold_triplet[1]
                    and isinstance(triplet[2], str)
                    and isinstance(gold_triplet[2], str)
                ):
                    # Calculate superposition as the length of the intersection over the length of the union
                    set_pred = set(triplet[2].split())
                    set_gold = set(gold_triplet[2].split())
                    if set_pred and set_gold:
                        intersection = set_pred & set_gold
                        union = set_pred | set_gold
                        score = len(intersection) / len(union)
                        if score > 0:
                            correct += score
                            break

    precision = correct / len(prediction) if prediction else 0
    recall = correct / len(gold) if gold else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1