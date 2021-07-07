# 파인튜닝 모델 평가
from sent_eval import BERTEmbeddingEvaluator

model = BERTEmbeddingEvaluator(model_fname = "./tune-ckpt", 
                               bertconfig_fname="./multi_cased_L-12_H-768_A-12/bert_config.json",
                               vocab_fname="./multi_cased_L-12_H-768_A-12/vocab.txt",
                               max_seq_length=32, dimension=768, num_labels=2, use_notebook=True)
