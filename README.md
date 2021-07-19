# NLP(Natural Language Procesing) examples (Chatbot)

본 예제는 자연어처리를 활용한 챗봇을 구현한 예제로서,    
샘플 데이터셋은 송영숙(songys)님이 제공해준 Chatbot_Data을 활용하였습니다.   

- 데이터셋 : [Chatbot_data(https://github.com/songys/Chatbot_data)](https://github.com/songys/Chatbot_data)
- 소스 참고 
    - 트랜스포머를 이용한 한국어 챗봇(Transformer Chatbot Tutorial) : [https://wikidocs.net/89786](https://wikidocs.net/89786) <br>
    - 텐서플로와 머신러닝으로 시작하는 자연어처리 : [https://github.com/NLP-kr/tensorflow-ml-nlp](https://github.com/NLP-kr/tensorflow-ml-nlp) <br>
    - 한국어 임베딩 : [https://github.com/ratsgo/embedding](https://github.com/ratsgo/embedding) <br>
    - 심리상담 챗봇 구현하기 : [https://rogerheederer.github.io/ChatBot_Wellness/](https://rogerheederer.github.io/ChatBot_Wellness/) <br>

###  학습모델 다운로드 
- BERT 분류 모델(NSMC) : [kobert_tune-ckpt.zip](http://jamjoong.org/jwlee/kobert_tune-ckpt.zip) <br>
    - Google BERT pretrained + fine tuning w/ NSMC dataset
- GPT2 챗봇 모델(일상대화) : [kogpt2-chatbot-dialog.pth](http://jamjoong.org/jwlee/kogpt2-chatbot-dialog.pth) <br>
    - GPT2 pretraining w/ Chatbot dataset
- GPT2 챗봇 모델(건강상담) : [kogpt2-chatbot-dialog_wellness.pth](http://jamjoong.org/jwlee/kogpt2-chatbot-dialog_wellness.pth) <br>
    - GPT2 pretraining w/ Wellness dataset

예제는 아래와 같이 구성되어 있습니다.

### 예제 구성
1. BERT 를 이용한 분류
    - 모델 훈련 : [01_bert_train.py](https://github.com/rightlit/nlp2/blob/main/examples/01_bert_train.py)
      - Google BERT 모델 다운로드, NSMC 데이터셋 전처리 후 구동 (01_bert_train.ipynb 참고)
      - BERT 기본 모델 : https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
      - NSMC 데이터셋 : https://github.com/e9t/nsmc
    - 모델 시험 : [01_bert_test.py](https://github.com/rightlit/nlp2/blob/main/examples/01_bert_test.py)
      - 학습모델 다운로드 후 구동
2. GPT2를 이용한 챗봇(일반대화)
    - 모델 훈련 : [02_chatbot_kogpt2_train.py](https://github.com/rightlit/nlp2/blob/main/examples/02_chatbot_kogpt2_train.py)
      - Chatbot 데이터셋 전처리 후 구동 (02_chatbot_kogpt2_train.ipynb 참고)
      - Chatbot 데이터셋 : https://github.com/songys/Chatbot_data
    - 모델 시험 : [02_chatbot_kogpt2_test.py](https://github.com/rightlit/nlp2/blob/main/examples/02_chatbot_kogpt2_test.py)
      - 학습모델 다운로드 후 구동
3. GPT2를 이용한 챗봇(건강상담)
    - 모델 훈련 : [03_chatbot_kogpt2_train.py](https://github.com/rightlit/nlp2/blob/main/examples/03_chatbot_kogpt2_train.py)
      - Wellness 데이터셋 전처리 후 구동 (03_chatbot_kogpt2_train.ipynb 참고)
      - Wellness 데이터셋 : https://github.com/nawnoes/WellnessConversation-LanguageModel
    - 모델 시험 : [03_chatbot_kogpt2_test.py](https://github.com/rightlit/nlp2/blob/main/examples/03_chatbot_kogpt2_test.py)
      - 학습모델 다운로드 후 구동
- - -
* 트랜스포머(transformers)를 이용한 챗봇 
    - 모델 훈련 : [11_chatbot_transformer_train.py](https://github.com/rightlit/nlp2/blob/main/examples/11_chatbot_transformer_train.py)
    - 모델 시험 : [11_chatbot_transformer_test.py](https://github.com/rightlit/nlp2/blob/main/examples/11_chatbot_transformer_test.py)

