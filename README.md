# NLP(Natural Language Procesing) examples

본 예제는 자연어처리의 기본 지식을 학습하는 예제로서,    
자연어처리 중 분류(Classification)를 주제로 구성하였으며,    
샘플 데이터셋은 네이버 영화감성 분석 데이터셋(Naver Movie Review Sentiment Analysis)을 활용하였습니다.   

- 데이터셋 : [NSMC 데이터셋(https://github.com/e9t/nsmc/)](https://github.com/e9t/nsmc/)
- 소스 참고 
    - 네이버 영화 리뷰 감성 분류하기(Naver Movie Review Sentiment Analysis) : [https://wikidocs.net/44249](https://wikidocs.net/44249) <br>
    - 1D CNN으로 IMDB 리뷰 분류하기 : [https://wikidocs.net/80783](https://wikidocs.net/80783) <br>
    - [Chapter 4. 분류] XGBoost(eXtraGradient Boost) : [https://injo.tistory.com/44](https://injo.tistory.com/44) <br>
    - 사전 훈련된 워드 임베딩을 이용한 의도 분류(Intent Classification using Pre-trained Word Embedding) : [https://wikidocs.net/86083](https://wikidocs.net/86083) <br>
    - 영어/한국어 Word2Vec 실습 : [https://wikidocs.net/50739](https://wikidocs.net/50739) <br>
    - 양방향 LSTM과 어텐션 메커니즘(BiLSTM with Attention mechanism) : [https://wikidocs.net/48920](https://wikidocs.net/48920) <br>

예제는 아래와 같이 구성되어 있습니다.

### 예제 구성
1. RNN 알고리즘을 이용한 분류 
    - 모델 훈련 : [01_xgboost_train.py](https://github.com/rightlit/nlp/blob/main/examples/01_xgboost_train.py)
    - 모델 평가 : [01_xgboost_eval.py](https://github.com/rightlit/nlp/blob/main/examples/01_xgboost_eval.py)
    - 모델 시험 : [01_xgboost_test.py](https://github.com/rightlit/nlp/blob/main/examples/01_xgboost_test.py)
2. RNN 알고리즘을 이용한 분류 
    - 모델 훈련 : [02_lstm_train.py](https://github.com/rightlit/nlp/blob/main/examples/02_lstm_train.py)
    - 모델 평가 : [02_lstm_eval.py](https://github.com/rightlit/nlp/blob/main/examples/02_lstm_eval.py)
    - 모델 시험 : [02_lstm_test.py](https://github.com/rightlit/nlp/blob/main/examples/02_lstm_test.py)
3. CNN 알고리즘을 이용한 분류 
    - 모델 훈련 : [03_cnn_train.py](https://github.com/rightlit/nlp/blob/main/examples/03_cnn_train.py)
    - 모델 평가 : [03_cnn_eval.py](https://github.com/rightlit/nlp/blob/main/examples/03_cnn_eval.py)
    - 모델 시험 : [03_cnn_test.py](https://github.com/rightlit/nlp/blob/main/examples/03_cnn_test.py)
4. 사전훈련된 Word2Vec 을 이용한 분류
    - 모델 훈련 : [04_cnn_word2vec_train.py](https://github.com/rightlit/nlp/blob/main/examples/04_cnn_word2vec_train.py)
    - 모델 평가 : [04_cnn_word2vec_eval.py](https://github.com/rightlit/nlp/blob/main/examples/04_cnn_word2vec_eval.py)
5. 직접훈련한 Word2Vec 을 이용한 분류
    - 모델 훈련 : [05_cnn_word2vec_train.py](https://github.com/rightlit/nlp/blob/main/examples/05_cnn_word2vec_train.py)
    - 모델 평가 : [05_cnn_word2vec_eval.py](https://github.com/rightlit/nlp/blob/main/examples/05_cnn_word2vec_eval.py)
6. Attention 적용 알고리즘을 이용한 분류 
    - 모델 훈련 : [06_lstm_attention_train.py](https://github.com/rightlit/nlp/blob/main/examples/06_lstm_attention_train.py)
    - 모델 평가 : [06_lstm_attention_eval.py](https://github.com/rightlit/nlp/blob/main/examples/06_lstm_attention_eval.py)
