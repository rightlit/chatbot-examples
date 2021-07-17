import sys, requests, random
sys.path.append('../models')

import tensorflow as tf
from bert.modeling import BertModel, BertConfig
from bert.tokenization import FullTokenizer, convert_to_unicode
#from bilm import Batcher, BidirectionalLanguageModel, weight_layers
#from preprocess import get_tokenizer, post_processing
#from supervised_nlputils import get_tokenizer, post_processing
from collections import defaultdict

import numpy as np
#from lxml import html
#from gensim.models import Doc2Vec, LdaModel
#from visualize_utils import visualize_homonym, visualize_between_sentences, \
#    visualize_self_attention_scores, visualize_sentences, visualize_words, visualize_between_words
#from tune_utils import make_elmo_graph, make_bert_graph
from tune_utils import make_bert_graph
#from sklearn.preprocessing import normalize


class SentenceEmbeddingEvaluator:

    def __init__(self, model_name, dimension, use_notebook=False):
        # reset graphs.
        tf.reset_default_graph()
        self.model_name = model_name
        self.dimension = dimension
        self.use_notebook = use_notebook

    def get_token_vector_sequence(self, sentence):
        raise NotImplementedError

    def get_sentence_vector(self, sentence):
        raise NotImplementedError

    def predict(self, sentence):
        raise NotImplementedError

    def tokenize(self, sentence):
        raise NotImplementedError

    def make_input(self, tokens):
        raise NotImplementedError

    def visualize_homonym(self, homonym, sentences, palette="Viridis256"):
        tokenized_sentences = []
        vecs = np.zeros((1, self.dimension))
        for sentence in sentences:
            tokens, vec = self.get_token_vector_sequence(sentence)
            tokenized_sentences.append(tokens)
            vecs = np.concatenate([vecs, vec], axis=0)
        visualize_homonym(homonym, tokenized_sentences, vecs, self.model_name, palette, use_notebook=self.use_notebook)

    def visualize_sentences(self, sentences, palette="Viridis256"):
        vecs = np.array([self.get_sentence_vector(sentence)[1] for sentence in sentences])
        visualize_sentences(vecs, sentences, palette, use_notebook=self.use_notebook)

    def visualize_between_sentences(self, sentences, palette="Viridis256"):
        vec_list = []
        for sentence in sentences:
            _, vec = self.get_sentence_vector(sentence)
            vec_list.append(vec)
        visualize_between_sentences(sentences, vec_list, palette, use_notebook=self.use_notebook)


class BERTEmbeddingEvaluator(SentenceEmbeddingEvaluator):

    def __init__(self, model_fname="/notebooks/embedding/data/sentence-embeddings/bert/tune-ckpt",
                 bertconfig_fname="/notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/bert_config.json",
                 vocab_fname="/notebooks/embedding/data/sentence-embeddings/bert/multi_cased_L-12_H-768_A-12/vocab.txt",
                 max_seq_length=32, dimension=768, num_labels=2, use_notebook=False):

        super().__init__("bert", dimension, use_notebook)
        config = BertConfig.from_json_file(bertconfig_fname)
        self.max_seq_length = max_seq_length
        self.tokenizer = FullTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        self.model, self.input_ids, self.input_mask, self.segment_ids, self.probs = make_bert_graph(config,
                                                                                                    max_seq_length,
                                                                                                    1.0,
                                                                                                    num_labels,
                                                                                                    tune=False)
        saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        checkpoint_path = tf.train.latest_checkpoint(model_fname)
        saver.restore(self.sess, checkpoint_path)

    def predict(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        probs = self.sess.run(self.probs, model_input)
        return probs

    """
    sentence를 입력하면 토크나이즈 결과와 token 벡터 시퀀스를 반환한다
        - shape :[[# of tokens], [batch size, max seq length, dimension]]
    """
    def get_token_vector_sequence(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        return [tokens, self.sess.run(self.model.get_sequence_output()[0], model_input)[:len(tokens) + 2]]

    """
    sentence를 입력하면 토크나이즈 결과와 [CLS] 벡터를 반환한다
         - shape :[[# of tokens], [batch size, dimension]]
    """
    def get_sentence_vector(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        return [tokens, self.sess.run(self.model.pooled_output, model_input)[0]]

    """
    sentence를 입력하면 토크나이즈 결과와 self-attention score matrix를 반환한다
        - shape :[[# of tokens], [batch size, # of tokens, # of tokens]]
    """
    def get_self_attention_score(self, sentence):
        tokens = self.tokenize(sentence)
        model_input = self.make_input(tokens)
        # raw_score : shape=[# of layers, batch_size, num_attention_heads, max_seq_length, max_seq_length]
        raw_score = self.sess.run(self.model.attn_probs_for_visualization_list, model_input)
        # 마지막 레이어를 취한 뒤, attention head 기준(axis=0)으로 sum
        scores = np.sum(raw_score[-1][0], axis=0)
        # scores matrix에서 토큰 개수만큼 취함
        scores = scores[:len(tokens), :len(tokens)]
        return [tokens, scores]

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(convert_to_unicode(sentence))

    def make_input(self, tokens):
        tokens = tokens[:(self.max_seq_length - 2)]
        token_sequence = ["[CLS]"] + tokens + ["[SEP]"]
        segment = [0] * len(token_sequence)
        sequence = self.tokenizer.convert_tokens_to_ids(token_sequence)
        current_length = len(sequence)
        padding_length = self.max_seq_length - current_length
        input_feed = {
            self.input_ids: np.array([sequence + [0] * padding_length]),
            self.segment_ids: np.array([segment + [0] * padding_length]),
            self.input_mask: np.array([[1] * current_length + [0] * padding_length])
        }
        return input_feed

    def visualize_self_attention_scores(self, sentence):
        tokens, scores = self.get_self_attention_score(sentence)
        visualize_self_attention_scores(tokens, scores, use_notebook=self.use_notebook)

