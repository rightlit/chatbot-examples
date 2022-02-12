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
from rank_bm25 import BM25Okapi
from scipy import sparse

# 클래스 선언 
class QnaSearch:

    # 속성 생성 
    def __init__(self): 
        #self.tokenizer = tokenizer 
        self.rawdata = [] 
        self.rawdata_q = []
        self.X_question = []
        self.features = []
        self.vectorize = None
        # BM25
        self.bm25 = None
        self.b = 0.75
        self.k1 = 1.6
        self.X_ = []
        self.avdl = None

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
        for i, row in enumerate(self.rawdata):
            if(i < 5):
                print(row)

        #return rawdata, rawdata_q

    # TF-IDF 벡터화
    def get_tfid_vector(self):

        #rawdata = corpus_data
        #rawdata_q = question_data
	
        ## TfidfVectorizer  방식으로 가중치를 주어서 Bow 를 만들다 
        self.vectorize = TfidfVectorizer(
            tokenizer=tokenizer,
            min_df=2,
            max_features=1000, #2048
            sublinear_tf=True    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
        )
        X = self.vectorize.fit_transform(self.rawdata)

        #new_rawdata = []
        #for row in rawdata:
        #    new_rawdata.append(text_cleaning(row))
        #X = vectorize.fit_transform(new_rawdata)
    
        print('fit_transform, (sentence {}, feature {})'.format(X.shape[0], X.shape[1]))    

        # 문장에서 뽑아낸 feature 들의 배열
        self.features = self.vectorize.get_feature_names()
        #print(features)

        #df_tfi = pd.DataFrame(X.toarray(), columns=self.features)
        #print(df_tfi)

        # 답변 문장으로 학습하고, 질문으로 유사도 비교
        # transform 만 수행

        print('TF-IDF vectorizing...')
        #X_question = vectorize.fit_transform(self.rawdata)
        self.X_question = self.vectorize.transform(self.rawdata_q)

        #return vectorize, X_question, features
	
	
    # 유사문장 검색
    def search_query(self, query_str): 
        #query_str = '마이너스 통장 신청하려고 합니다'
        #features = key_features
        #vectorize = vec
    
        srch=[t for t in tokenizer(query_str) if t in self.features]
        #print(srch)
        file_log(" ".join(srch))
    
        # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
        srch_dtm = np.asarray(self.X_question.toarray())[:, [
            # vectorize.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
            self.vectorize.vocabulary_.get(i) for i in srch
        ]]

        score = srch_dtm.sum(axis=1)
        #print(score)
        file_log('score:' + str(score))
    
        #rawdata_q = question_data
        cnt = 0
        ret_str = ''
        ret_str = '[Q]' + ' '.join(srch)
        for i in score.argsort()[::-1]:
            if score[i] > 0:
                #print('{} / score : {}'.format(rawdata_q[i], score[i]))
                file_log('{} / score : {}'.format(self.rawdata_q[i], score[i]))
            ret_str = ret_str  + '|' + '{} / score : {}'.format(self.rawdata_q[i], score[i])

            cnt = cnt + 1
            if(cnt > 10):
                break

        print('ret_str : ', ret_str)
        file_log(ret_str)
        return ret_str
 
    def _get_bm25_vector(self):
        corpus = self.rawdata_q
        tokenized_corpus = [tokenizer(doc) for doc in corpus]
        #srch=[t for t in tokenizer(input_text) if t in features]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _search_query_bm25(self, query_str):
        #query = "windy London"
        #query = '마이너스 통장 신청하려고 합니다'
        query = query_str
        #tokenized_query = query.split(" ")
        tokenized_query = tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        #print(doc_scores) # array type

        corpus = self.rawdata_q
        rs_list = self.bm25.get_top_n(tokenized_query, corpus, n=10)
        ret_str = '|'.join(rs_list)
        return ret_str

    # BM25 벡터화
    def get_bm25_tfid_vector(self):
        self.vectorize = TfidfVectorizer(
            tokenizer=tokenizer,
            min_df=2,
            max_features=1000, #2048
            sublinear_tf=True    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
        )
        self.vectorize.fit(self.rawdata_q)
        #y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.X_ = super(TfidfVectorizer, self.vectorize).transform(self.rawdata_q)
        self.avdl = self.X_.sum(1).mean()
 
    def bm25_tfid_transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        #X = super(TfidfVectorizer, self.vectorizer).transform(X)
        X = self.X_
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorize).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1

    # 유사문장 검색(BM25)
    def search_query_bm25(self, query_str):
        #X_score = self.bm25.transform(query_str, self.rawdata_q)
        X_score = self.bm25_tfid_transform(query_str)

        score = X_score
        cnt = 0
        ret_str = ''
        ret_str = '[Q]' + query_str
        for i in score.argsort()[::-1]:
            if score[i] > 0:
                #print('{} / score : {}'.format(rawdata_q[i], score[i]))
                file_log('{} / score : {}'.format(self.rawdata_q[i], score[i]))
            ret_str = ret_str  + '|' + '{} / score : {}'.format(self.rawdata_q[i], score[i])

            cnt = cnt + 1
            if(cnt > 10):
                break

        print('ret_str : ', ret_str)
        file_log(ret_str)

        return ret_str

# 인스턴스 생성 
#qna = QnaSearch() 
# 메소드 호출 
#qna.load_corpus_data('/content/chatbot_faq_all.txt_new')
