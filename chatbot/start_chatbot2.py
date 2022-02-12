#!/usr/local/bin/python

##### main
from flask import Flask
app = Flask(__name__)

#from corpus_util import load_corpus_data, get_tfid_vector, tokenizer
from corpus_util import file_log
from QnaSearch import QnaSearch
import numpy as np

qna = QnaSearch()
qna.load_corpus_data('/content/chatbot_faq_all.txt_new')
#qna.get_tfid_vector()
#ret_str = qna.search_query('대출연장')

qna.get_bm25_tfid_vector()
#ret_str = qna.search_query_bm25('대출연장')

#corpus_data, question_data = load_corpus_data('/content/chatbot_faq_all.txt_new')
#vec, X_question, key_features = get_tfid_vector(corpus_data, question_data)

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
    #ret_str = qna.search_query(query_str)
    ret_str = qna.search_query_bm25(query_str)
    
    return ret_str

# ready
print('Chatbot Flask daemon Ready!!!')
file_log('Chatbot Flask daemon Ready!!!')

# APP 데몬 시작
app.run(host="localhost",port=5001)
