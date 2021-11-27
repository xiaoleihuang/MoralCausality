import numpy as np
from loguru import logger

def F1(prediction,target,epoch):
    prediction = np.array((np.array(prediction)>0.5),dtype=float)
    target = np.array(target)
    TP = (prediction*target).sum()
    precision = TP/prediction.sum()
    recall = TP/target.sum()
    f1 = 2 * (precision * recall)/(precision+recall)
    logger.info('best test epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, f1, precision, recall))
    return precision, recall, f1