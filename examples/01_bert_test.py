import numpy as np

# 파인튜닝 모델 평가
from sent_eval import BERTEmbeddingEvaluator

model = BERTEmbeddingEvaluator(model_fname = "./tune-ckpt", 
                               bertconfig_fname="./multi_cased_L-12_H-768_A-12/bert_config.json",
                               vocab_fname="./multi_cased_L-12_H-768_A-12/vocab.txt",
                               max_seq_length=32, dimension=768, num_labels=2, use_notebook=True)
                               
def sentiment_predict(sentence):
  p_data = model.predict(sentence) # 예측
  p_max = np.max(p_data)
  score = p_max 
  if(score > 0.5):
    print("{} : 긍정 리뷰입니다.{:.1%}\n".format(sentence, p_max))
  else:
    print("{} : 부정 리뷰입니다.{:.1%}\n".format(sentence, p_max))

# input your words
input_data = ['영화 짱 재밌어요!',
              '완전 재미없어요 ㅠ',
              '완전 비추입니다 ㅎㅎ',
              '대박 최고 영화에요!',
              '배우들 연기잘하고 감동적임다',
              '이 영화 개꿀잼 ㅋㅋㅋ', 
              '이 영화 핵노잼 ㅠㅠ', 
              '이딴게 영화냐 ㅉㅉ', 
              '감독 뭐하는 놈이냐?', 
              '와 개쩐다 정말 세계관 최강자들의 영화다']

for s in input_data:
    sentiment_predict(s)
