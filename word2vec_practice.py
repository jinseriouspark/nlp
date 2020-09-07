# 출처 : https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72

import numpy as np
import re

def tokenize(text): # 단어를 토큰화
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\'*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower()) # 모든 단어 소문자화

def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y= [], []

    for i in range(N): # 토큰의 길이만큼 반복
        print(i)
        nbr_inds = list(range(max(0, i - window_size), i)) + \
            list (range(i + 1, min(N, i + window_size + 1)))
            # nbr_inds: 한 문장의 단어를 모두 훑으면서 window_size 만큼 왼쪽 오른쪽에 있는 단어를 긁어옴
        for j in nbr_inds:
            #print(tokens[i], tokens[j])
            X.append(word_to_id[tokens[i]]) #X 에는 i, 입력하는 값이 들어가고
            Y.append(word_to_id[tokens[j]]) #Y 에는 i 주변의 window_size 안의 단어들이 들어옴
        #print('X is', X, 'Y is', Y)
    X = np.array(X)
    X = np.expand_dims(X, axis = 0) # 원래 차원에다 행 차원 1개 더 만들어줌
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis = 0)

    return X, Y

doc = "I ordered pig front foot for night snack.\
 Yesterday I made a plan to loose my weight but it couldn't stop ordering food"
tokens = tokenize(doc)
word_to_id, id_to_word = mapping(tokens)
X, Y = generate_training_data(tokens, word_to_id, 3)
vocab_size = len(id_to_word)
m = Y.shape[1] # window_size 에 속한 단어 갯수
# Y 를 원핫인코딩으로 변형
Y_one_hot = np.zeros((vocab_size, m)) # id_to_word 에 포함된 단어 x Y 길

