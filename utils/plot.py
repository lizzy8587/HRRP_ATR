import torch



def confusion_matrix(preds, labels, cls_num):
    conf_matrix = torch.zeros(cls_num, cls_num)
    if len(preds.shape) > 1:
        preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix