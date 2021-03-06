import os
import pandas as pd
import re

# 파일 로깅
def file_log(s):
    f = open('/tmp/flask.log', 'a')
    f.write(str(s) + '\n')
    f.close()

# 텍스트 정제
def text_cleaning(text): 
        #이메일 주소 제거
        email =re.compile('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        text = email.sub('', text) 
        #URL 제거
        url =re.compile('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
        text = url.sub('', text) 
        #HTML 제거
        html =re.compile('<[^>]*>')
        text = html.sub('', text) 

        #특수문자를 공백으로 대체(문장을 살리기위헤 마침표는 남겨둠)
        #special =re.compile('[^\w\s]')
        #text = special.sub(' ', text) 
        special= ['*', '{', ',', ':', ']', '$', '+', '[', '#', '(', '%', '&', '}', '`', ''', ''','·',
                    '=', ';', '>','＞', '/', '"', '"', '"', '\\', '?', '~', "'", '<', ')', '^', '!', '_',
                    '|', '@','@','©','ⓒ', '℗','®','①', '-','▶','…','☞','▲','◆','■', #'.', 빼고
                    '☎', '※','②','③','④'] 
        for chr in special :
            text=text.replace(chr,' ')

            #특수문자 제거 후 생기는 중복된 공백 제거
            while text.find('  ') > 0:
                text = text.replace('  ',' ' ) # 중복된 공백 제거

            #특수문자 제거 후 생기는 중복된 개행 제거
            while text.find('\n\n') > 0:
                text = text.replace('\n\n','\n' ) # 중복된 개행 제거

            #좌우측 공백 삭제
            text = text.strip()
        return text

def load_corpus_data(data_file):

    ##### FAQ DATA LOAD
    print('FAQ DATA LOAD loading...')
    faq_file = data_file
    #faq_file = '/content/chatbot_faq_all.txt_new'
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
	#retuern json_list

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
    for i, row in enumerate(rawdata):
        if(i < 5):
            print(row)

    return rawdata, rawdata_q

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

def get_tfid_vector(corpus_data, question_data):

    rawdata = corpus_data
    rawdata_q = question_data
	
    ## TfidfVectorizer  방식으로 가중치를 주어서 Bow 를 만들다 
    vectorize = TfidfVectorizer(
        tokenizer=tokenizer,
        min_df=2,
        max_features=1000, #2048
        sublinear_tf=True    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
    )
    X = vectorize.fit_transform(rawdata)

    #new_rawdata = []
    #for row in rawdata:
    #    new_rawdata.append(text_cleaning(row))
    #X = vectorize.fit_transform(new_rawdata)
    
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

    print('TF-IDF vectorizing...')
    #X_question = vectorize.fit_transform(rawdata)
    X_question = vectorize.transform(rawdata_q)

    return vectorize, X_question, features
