# 유틸함수
from corpus_util import tokenizer, file_log

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
np.random.seed(0)  # 랜덤 난수를 지정하여 사용

from konlpy.tag import Twitter
twitter = Twitter()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity 

# 클래스 선언 
class QnaSearch:

    # 속성 생성 
    def __init__(self): 
        #self.tokenizer = tokenizer 
        self.rawdata = [] 
        self.rawdata_q = []
        self.vectorize = None

    # CORPUS 데이터 불러오기
    def load_corpus_data(self, data_file):
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
        #rawdata_q = []
        rownum = 0

        for json_item in json_list:
            rawdata_dic[str(json_item['answer']).strip()] = rownum
            self.rawdata_q.append(str(json_item['question']))
            rownum = rownum +1
    
        #rawdata = []
        self.rawdata = list(rawdata_dic.keys())
        for i, row in enumerate(rawdata):
            if(i < 5):
                print(row)

        #return rawdata, rawdata_q
        
    # 유사문장 검색
    def search_query(self, query_str): 
        print('')

    def query(query_str):
        #query_str = '마이너스 통장 신청하려고 합니다'
        features = key_features
        vectorize = vec
    
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
        
       
# 인스턴스 생성 
qna = QnaSearch('/content/chatbot_faq_all.txt_new') 
# 메소드 호출 
qna.load_corpus_data()
