#coding:utf-8

import sys
import math
import random
from collections import defaultdict
from sklearn.cross_validation import train_test_split

def NV_train(X_train, Y_train):
    
    categories = defaultdict(lambda: 0)
    word_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for x, y in zip(X_train, Y_train):
        categories[y] += 1
        word_features = x[:54]
        other_features = " ".join(map(str, [int(round(a,(len(str(int(a)))-1)*-1)) if (i>0) else int(a) for i, a in enumerate(x[54:])]))
        for w_id, d in enumerate(word_features):
            word_dict[y][w_id] += d
        word_dict[y][other_features] += 1

    #p_catの計算
    p_cates = {c:float(v)/len(Y_train) for c, v in categories.items()}

    #各カテゴリの全単語数
    numwords_cates = {c: sum(cate_dict.values()) for c, cate_dict in word_dict.items()}    

    #全ユニーク単語数
    V = len(list(set([c for cate_dict in word_dict.values() for c in cate_dict])))

    return word_dict, p_cates, numwords_cates, V

def NV_predict(word_dict, p_cates, numwords_cates, V, X_test, Y_test):

    answer_lis=[]
    predict_lis=[]
    for data, answer in zip(X_test, Y_test):
        scores={}
        word_features = data[:54]
        other_features = " ".join(map(str, [int(round(a,(len(str(int(a)))-1)*-1)) if (i>0) else int(a) for i, a in enumerate(data[54:])]))
        for cate, p in p_cates.items():
            p = math.log(p)
            p_cate_word = 0
            for i, feature in enumerate(word_features):
                p_cate_word += math.log(((float(word_dict[cate][i] + 1))/(numwords_cates[cate] + V)))*feature #+1, +Vはスムージングのため.
            p_cate_word += math.log((float(word_dict[cate][other_features] + 1))/(numwords_cates[cate] + V))
            scores[cate] = p + p_cate_word
        predict=max(scores.items(), key=lambda x:x[1])[0]
        print "Answer=%s, Predict=%s" % (answer, predict)
        answer_lis.append(answer)
        predict_lis.append(predict)

    correct=0
    for a, p in zip(answer_lis, predict_lis):
        if(a==p):
            correct+=1
    accuracy = float(correct)/len(answer_lis)
    
    return accuracy

def read_data(f):
    X=[]
    Y=[]
    for line in f:
        data=line.split(",")
        X.append([float(a) for a in data[:57]])
        Y.append(int(data[-1]))
        
    return (X, Y)

if __name__ == "__main__":
    
    with open("./spambase.data", "r") as f:
        X, Y = read_data(f)

    #交差検定
    n_fold=10
    accuracys=[]
    for i in range(n_fold):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
        word_dict, p_cates, numwords_cates, V = NV_train(X_train, Y_train)
        print "====%d====" % (i)
        print "trining finished"
        accuracy = NV_predict(word_dict, p_cates, numwords_cates, V, X_test, Y_test)    
        accuracys.append(accuracy)
        print "accuracy=%f" % (accuracy)
    print "===="
    print "average accuracy=%f" % (sum(accuracys)/len(accuracys))