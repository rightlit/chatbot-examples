#!/usr/local/bin/python


##### FAQ DATA LOAD
import pandas as pd
faq_file = '/content/chatbot_faq_all.txt_new'
colnames = ['ID', '질문', '답변', '대분류ID', '대분류', '중분류ID', '중분류', '소분류ID', '소분류', '사용여부', '연결 시나리오ID', '연결 시나리오명', '웹 링크 버튼', '웹 링크 URL', '대화모형 사용여부', '시나리오 사용여부', '작성자ID', '작성자 로그인 ID', '작성자', '작성일', '수정자ID', '수정자 로그인 ID', '수정자', '최종수정일', '상담 ID', '메세지 ID']
selected_cols = ['ID', '질문', '답변', '대분류', '중분류', '소분류']
df_faq_data = pd.read_csv(faq_file, sep='\t', encoding='cp949', names=colnames, usecols=selected_cols)
#print(df_faq_data)

##### dataframe to json list
json_list = []
df_faq_id = df_faq_data['ID'].tolist()
df_faq_question = df_faq_data['질문'].tolist()
df_faq_answer = df_faq_data['답변'].tolist()
df_faq_lcd = df_faq_data['대분류'].tolist()
df_faq_mcd = df_faq_data['중분류'].tolist()
df_faq_scd = df_faq_data['소분류'].tolist()

for i in range(len(df_faq_id)):
    json_item = {}
    json_item['ID'] = df_faq_id[i]
    json_item['question'] = df_faq_question[i]
    json_item['answer'] = df_faq_answer[i]
    json_item['lcd'] = str(df_faq_lcd[i]).strip()  # trim
    json_item['mcd'] = str(df_faq_mcd[i]).strip()  # trim
    json_item['scd'] = str(df_faq_scd[i]).strip()  # trim
    json_list.append(json_item)

#print(json_list)

##### 함수 정의
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
np.random.seed(0)  # 랜덤 난수를 지정하여 사용

from konlpy.tag import Twitter
twitter = Twitter()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity 

# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def tokenizer(raw, pos=["Noun","Alpha","Verb","Number"], stopword=[]):
    return [
        word for word, tag in twitter.pos(
            raw, 
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]

##### CORPUS 만들기 (rawdata)
# 질문/답변 데이터 분리
rawdata_dic = {}
rawdata_q = []
answer_prev = ''
rownum = 0

for json_item in json_list:
    rawdata_dic[str(json_item['answer']).strip()] = rownum
    rawdata_q.append(str(json_item['question']))
    rownum = rownum +1
    
#rawdata = []
rawdata = list(rawdata_dic.keys())
for row in rawdata:
    print(row)

## TfidfVectorizer  방식으로 가중치를 주어서 Bow 를 만들다 
vectorize = TfidfVectorizer(
    tokenizer=tokenizer,
    min_df=2,
    max_features=1000, #2048
    sublinear_tf=True    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
)
#X = vectorize.fit_transform(rawdata)
X = vectorize.fit_transform(new_rawdata)

print(
    'fit_transform, (sentence {}, feature {})'.format(X.shape[0], X.shape[1])
)    

# 문장에서 뽑아낸 feature 들의 배열
features = vectorize.get_feature_names()
#print(features)

df_tfi = pd.DataFrame(X.toarray(), columns=features)
#print(df_tfi)

# 답변 문장으로 학습하고, 질문으로 유사도 비교
# transform 만 수행

#X_question = vectorize.fit_transform(rawdata)
X_question = vectorize.transform(rawdata_q)


##### main
from flask import Flask
app = Flask(__name__)

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
def query(page):
    #query_str = '마이너스 통장 신청하려고 합니다'
    srch=[t for t in tokenizer(query_str) if t in features]
    #print(srch)
    
    # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
    srch_dtm = np.asarray(X_question.toarray())[:, [
        # vectorize.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
        vectorize.vocabulary_.get(i) for i in srch
    ]]

    score = srch_dtm.sum(axis=1)
    #print(score)
    
    cnt = 0
    for i in score.argsort()[::-1]:
        if score[i] > 0:
            print('{} / score : {}'.format(rawdata_q[i], score[i]))
        cnt = cnt + 1
        if(cnt > 10):
            break
    return srch


# APP 데몬 시작
app.run(host="localhost",port=5001)
