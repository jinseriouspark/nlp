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

# 지금까지 단어 전처리를 진행
# 이제는 가중치를 초기화 하고 순전파 역전파를 준비하면서, 손실계산을 한 뒤 gradient업데이트까지 진행해야 한다
# 우선 문장 1개 = 단어 x 임베딩 차원으로 변환하는 것이 필요

def initialize_wrd_emb(vocab_size, emb_size):
    """
    vocab_size : 학습하고자 하는 데이터나 코퍼스 내 단어 갯수
    emb_size : 초기 단어 임베딩 데이터. 각 단어를 몇개의 차원으로 임베딩할 것인지 정해야 함
    """
    wrd_emb = np.random.randn(vocab_size, emb_size) * 0.01
    return wrd_emb

def initialize_dense(input_size, output_size):
    """
    :param input_size: dense_layer 의 input 크기
    :param output_size: dense_layer 의 ouput 크기
    :return: 가중치 행렬
    """
    W = np.random.randn(output_size, input_size) * 0.01
    return W

def initalize_parameters(vocab_size, emb_size):
    """
    :param vocab_size: 파라미터 학습에 필요한 단어 크기
    :param emb_size: 파라미터 학습에 필요한 임베딩 사이즈 크기
    :return: 파라미터
    """
    wrd_emb = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)

    parameters = {} # 파라미터를 받을 사전 하나 준비
    parameters['wrd_emb'] = wrd_emb # 사전에는 워드 임베딩 정보와
    parameters['W'] = W # 가중치 정보를 넣어둠
    return parameters

# 방법
## 문장 1개가 들어오면 임베딩 차원만큼 열을 만들어 단어 행렬을 만듦
## 그리고 은닉층(여기에서는 dense layer) 에 넣은 후
## 활성함수 중 softmax 함수를 사용하여 결과물을 만들게 된다


def ind_to_word_vecs(inds, parameters):
    '''
    :param inds: 넘파이 배열이고 1x m 크기
    :param parameters: 딕셔너리 형태 학습된 가중치 저장소
    :return:
    '''
    m = inds.shape[1]
    wrd_emb = parameters['wrd_emb']
    word_vec = wrd_emb[inds.flatten(), :]. T
    assert(word_vec.shape == (wrd_emb.shape[1], m))
    # assert() 는 맞을 때 정상적으로 동작 틀리면 assertion error 가 발생한다
    return word_vec

def linear_dense(word_vec, parameters):
    """
    :param word_vec: 임베딩 사이즈 x m
    :param parameters: 사전형, 학습된 가중치 저장
    :return:
    """
    m = word_vec.shape[1]
    w = parameters['w']
    z = np.dot(w, word_vec)

    return w, z

def softmax(z):
    """
    :param z: dense layer 의 결과물 vocab_size , m 으로 구성
    :return:
    """
    softmax_out = np.divide(np.exp(z), np.sum(np.exp(x), axis = 0, keepdims = True) + 0.001)
    return softmax_out

def forward_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    w, z = linear_dense(word_vec, parameters)
    softmax_out = softmax(z)

    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['w'] = w
    caches['z'] = z
    return softmax_out, caches

def cross_entropy(softmax_out, y):
    """
    :param softmax_out: 소프트 맥스 결과물, vocab_size, m
    :param y:
    :return:
    """
    m = softmax_out.shape[1]
    cost = -(1/m) * np.sum(np.sum(y * np.log(softmax_out + 0.001), \
                                  axis = 0,keepdims=True ),axis = 1 )
    return cost

# 역천파 과정에서 각각의 손실함수가 반영된 학습가능한 가중치의 gradient 를 계산하고자 하고
# 그 그라디언트로 가중치를 업데이트 할 것이다
# 내적이 된 것들은 (아마) 역행렬을 곱하게 되고
# 더해진 것들은 그대로 타인이 반영이 되고 그럴텐데, 정확하게 코드를 통해 다시 확인해보자

def softmax_backward(Y, softmax_out):
    '''
    :param Y: 학습 데이터의 라벨 vocab_size x m
    :param softmax_out: 소프트맥스 결과물
    :return:
    '''
    dl_dz = softmax_out - Y # 예측 결과 - 실제 결과
    return dl_dz

def dense_backward(dl_dz, caches):
    """
    :param dl_dz: vocab_size x m 크기로 생김
    :param caches: 사전형태. 순전파의 각 단계 결과들
    :return:
    """
    w = caches['w']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]

    dl_dw = (1 / m) * np.dot(dl_dz, word_vec.T)
    # 역전파는 역행렬을 내적한 뒤 평균(1/m) 을 낸다
    dl_dword_vec = np.dot(w.T, dl_dz)
    return dl_dw, dl_dword_vec

def backward_propagation(Y, softmax_out, caches):
    dl_dz = softmax_backward(Y, softmax_out)
    dl_dw, dl_dword_vec = dense_backward(dl_dz, caches)
    gradients = dict()
    gradients['dl_dz'] = dl_dz
    gradients['dl_dw'] = dl_dw
    gradients['dl_dword_vec'] = dl_dword_vec

    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = parameters['wrd_emb'].shape
    inds = caches['inds']
    wrd_emb = parameters['wrd_emb']
    dl_dword_vec = gradients['dl_dword_vec']
    m = inds.shape[-1]

    wrd_emb[inds.flatten(), :] -= dl_dword_vec.T * learning_rate
    parameters['w'] -= learning_rate *gradients['dl_dw']

# 모델 학습을 위해서 순, 역전파
# 스킵그램 모듈

def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None,
                            print_cost=True, plot_cost=True):
    """
    X: Input word indices. shape: (1, m)
    Y: 원핫인코딩된 결과. shape: (vocab_size, m)
    vocab_size: 학습 시 단어
    emb_size: 임베딩 크
    learning_rate: 학습률
    epochs: 에폭 (학습 몇 번 반복할 것인가)
    batch_size: 미니 배치 사이즈 크기
    parameters: 미리 학습 및 초기화 파라미
    """
    costs = []
    m = X.shape[1]

    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size) # 파라미터를 우선 만듦

    for epoch in range(epochs):
        epoch_cost = 0 # 에폭 별 비용
        batch_inds = list(range(0, m, batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds: # 배치크기에 따라 학습 데이터 잘라준다
            X_batch = X[:, i:i + batch_size]
            Y_batch = Y[:, i:i + batch_size]

            softmax_out, caches = forward_propagation(X_batch, parameters) #순전파 후 소프트맥스 함수에 넣을 것과 캐쉬
            gradients = backward_propagation(Y_batch, softmax_out, caches) # y 배치값 소프트맥수 함수 캐시를 가지고 그라디언트 계산
            update_parameters(parameters, caches, gradients, learning_rate) # 파라미터와 캐시 그라디언트 학습률로 파라미터 업데이트
            cost = cross_entropy(softmax_out, Y_batch) # 크로스 엔트로피를 사용해서 비용 계산
            epoch_cost += np.squeeze(cost) # 에폭 별 비용에 누적 합산

        costs.append(epoch_cost) # 에폭 비용을 총 비용에 추가
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98 # 학습률을 낮춰줆

    if plot_cost: # 그림그려주는 부분
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
    return parameters