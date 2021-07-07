# fine tuning (num_labels = 2)
!python ./tune_utils.py --model_name bert \
  --train_corpus_fname ./train.txt.1 \
  --test_corpus_fname ./test.txt.1 \
  --vocab_fname ./multi_cased_L-12_H-768_A-12/vocab.txt \
  --pretrain_model_fname ./multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --config_fname ./multi_cased_L-12_H-768_A-12/bert_config.json \
  --model_save_path ./tune-ckpt \
  --num_labels 2