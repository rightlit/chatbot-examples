#!/usr/local/bin/python

##### main
from flask import Flask
app = Flask(__name__)

from corpus_util import load_corpus_data, get_tfid_vector

def filg_log(s):
    f = open('./flask.log', 'a')
    f.write(str(s) + '\n')
    f.close()

corpus_data, question_data = load_corpus_data('/content/chatbot_faq_all.txt_new')
X_question = get_tfid_vector(corpus_data, question_data)

# HELLO
@app.route('/hello')
def board():
    return "Hello, World"

# URL parameter
@app.route('/board/<article_idx>')
def board_view(article_idx):
    return '게시판 번호: ' + str(article_idx)

# URL parameter(default)
@app.route('/boards',defaults={'page':'index'})
@app.route('/boards/<page>')
def boards(page):
    return page+" 페이지입니다."

# QUERY sample
@app.route('/query/<query_str>')
def query(query_str):
    #query_str = '마이너스 통장 신청하려고 합니다'
    srch=[t for t in tokenizer(query_str) if t in features]
    #print(srch)
    file_log(" ".join(srch))
    
    # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
    srch_dtm = np.asarray(X_question.toarray())[:, [
        # vectorize.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
        vectorize.vocabulary_.get(i) for i in srch
    ]]

    score = srch_dtm.sum(axis=1)
    #print(score)
    file_log('score:' + str(score))
    
    rawdata_q = question_data
    cnt = 0
    ret_str = ''
    ret_str = '[Q]' + ' '.join(srch)
    for i in score.argsort()[::-1]:
        if score[i] > 0:
            #print('{} / score : {}'.format(rawdata_q[i], score[i]))
            file_log('{} / score : {}'.format(rawdata_q[i], score[i]))
        ret_str = ret_str  + '|' + '{} / score : {}'.format(rawdata_q[i], score[i])

        cnt = cnt + 1
        if(cnt > 10):
            break

    print('ret_str : ', ret_str)
    file_log(ret_str)
    return ret_str

# ready
print('Chatbot Flask daemon Ready!!!')
file_log('Chatbot Flask daemon Ready!!!')

# APP 데몬 시작
app.run(host="localhost",port=5001)
