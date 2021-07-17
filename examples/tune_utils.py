import sys, os, random, argparse, re, collections
import numpy as np
import tensorflow as tf
#from tensorflow.contrib import nccl
from tensorflow.python.ops import nccl_ops

#from gensim.models import Word2Vec
from collections import defaultdict
#from scipy.stats import truncnorm
#import sentencepiece as spm

sys.path.append('../models')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from preprocess import get_tokenizer, post_processing
from supervised_nlputils import get_tokenizer, post_processing
#from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer
from bert.tokenization import FullTokenizer, convert_to_unicode



def make_bert_graph(bert_config, max_seq_length, dropout_keep_prob_rate, num_labels, tune=False):
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='inputs_ids')
    input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
    model = BertModel(config=bert_config,
                      is_training=tune,
                      input_ids=input_ids,
                      input_mask=input_mask,
                      token_type_ids=segment_ids)

    if tune:
        output_layer = model.get_pooled_output()
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    else:
        output_layer = model.get_pooled_output()
        label_ids = None

    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    if tune:
        # loss layer
        one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return input_ids, input_mask, segment_ids, label_ids, logits, loss
    else:
        # prob layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return model, input_ids, input_mask, segment_ids, probs

'''
    if tune:
        bert_embeddings_dropout = tf.nn.dropout(model.pooled_output, keep_prob=(1 - dropout_keep_prob_rate))
        #bert_embeddings_dropout = tf.nn.dropout(model.pooled_output, keep_prob=0.9)
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    else:
        bert_embeddings_dropout = model.pooled_output
        label_ids = None


    logits = tf.contrib.layers.fully_connected(inputs=bert_embeddings_dropout,
                                               num_outputs=num_labels,
                                               activation_fn=None,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               biases_initializer=tf.zeros_initializer())
    if tune:
        # loss layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
        loss = tf.reduce_mean(CE)
        return input_ids, input_mask, segment_ids, label_ids, logits, loss
    else:
        # prob layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return model, input_ids, input_mask, segment_ids, probs
'''

def make_word_embedding_graph(num_labels, vocab_size, embedding_size, tune=False):
    ids_placeholder = tf.placeholder(tf.int32, [None, None], name="input_ids")
    input_lengths = tf.placeholder(tf.int32, [None], name="input_lengths")
    labels_placeholder = tf.placeholder(tf.int32, [None], name="label_ids")
    if tune:
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    else:
        dropout_keep_prob = tf.constant(1.0, dtype=tf.float32)
    We = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True)
    embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_size])
    embed_init = We.assign(embedding_placeholder)
    # shape : [batch_size, unroll_steps, dimension]
    embedded_words = tf.nn.embedding_lookup(We, ids_placeholder)
    # input of fine tuning network
    features = tf.nn.dropout(embedded_words, dropout_keep_prob)
    # Bidirectional LSTM Layer
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=embedding_size,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=embedding_size,
                                           cell_clip=5,
                                           proj_clip=5)
    lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                     cell_bw=lstm_cell_bw,
                                                     inputs=features,
                                                     sequence_length=input_lengths,
                                                     dtype=tf.float32)

    # Attention Layer
    output_fw, output_bw = lstm_output
    H = tf.contrib.layers.fully_connected(inputs=output_fw + output_bw, num_outputs=256, activation_fn=tf.nn.tanh)
    attention_score = tf.nn.softmax(tf.contrib.layers.fully_connected(inputs=H, num_outputs=1, activation_fn=None), axis=1)
    attention_output = tf.squeeze(tf.matmul(tf.transpose(H, perm=[0, 2, 1]), attention_score), axis=-1)
    layer_output = tf.nn.dropout(attention_output, dropout_keep_prob)

    # Feed-Forward Layer
    fc = tf.contrib.layers.fully_connected(inputs=layer_output,
                                           num_outputs=512,
                                           activation_fn=tf.nn.relu,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer())
    features_drop = tf.nn.dropout(fc, dropout_keep_prob)
    logits = tf.contrib.layers.fully_connected(inputs=features_drop,
                                               num_outputs=num_labels,
                                               activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer())
    if tune:
        # Loss Layer
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)
        loss = tf.reduce_mean(CE)
        return ids_placeholder, input_lengths, labels_placeholder, dropout_keep_prob, embedding_placeholder, embed_init, logits, loss
    else:
        # prob Layer
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return ids_placeholder, input_lengths, labels_placeholder, probs


class Tuner(object):

    def __init__(self, train_corpus_fname=None, tokenized_train_corpus_fname=None,
                 test_corpus_fname=None, tokenized_test_corpus_fname=None,
                 model_name="bert", model_save_path=None, vocab_fname=None, eval_every=1000,
                 batch_size=32, num_epochs=10, dropout_keep_prob_rate=0.9, model_ckpt_path=None,
                 sp_model_path=None, num_labels=2):
        # configurations
        tf.logging.set_verbosity(tf.logging.INFO)
        self.model_name = model_name
        self.eval_every = eval_every
        self.model_ckpt_path = model_ckpt_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_keep_prob_rate = dropout_keep_prob_rate
        self.best_valid_score = 0.0
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        self.num_labels = num_labels
        self.valid_scores = []

        # define tokenizer
        if self.model_name == "bert":
            self.tokenizer = FullTokenizer(vocab_file=vocab_fname, do_lower_case=False)
        elif self.model_name == "xlnet":
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model_path)
            self.tokenizer = sp
        else:
            self.tokenizer = get_tokenizer("mecab")
        # load or tokenize corpus
        self.train_data, self.train_data_size = self.load_or_tokenize_corpus(train_corpus_fname, tokenized_train_corpus_fname)
        self.test_data, self.test_data_size = self.load_or_tokenize_corpus(test_corpus_fname, tokenized_test_corpus_fname)

        self.num_batches_per_epoch = 0

    def load_or_tokenize_corpus(self, corpus_fname, tokenized_corpus_fname):
        data_set = []
        if os.path.exists(tokenized_corpus_fname):
            tf.logging.info("load tokenized corpus : " + tokenized_corpus_fname)
            with open(tokenized_corpus_fname, 'r') as f1:
                for line in f1:
                    tokens, label = line.strip().split("\u241E")
                    if len(tokens) > 0:
                        data_set.append([tokens.split(" "), int(label)])
        else:
            tf.logging.info("tokenize corpus : " + corpus_fname + " > " + tokenized_corpus_fname)
            with open(corpus_fname, 'r') as f2:
                next(f2)  # skip head line
                for line in f2:
                    sentence, label = line.strip().split("\u241E")
                    if self.model_name == "bert":
                        tokens = self.tokenizer.tokenize(sentence)
                    elif self.model_name == "xlnet":
                        normalized_sentence = preprocess_text(sentence, lower=False)
                        tokens = encode_pieces(self.tokenizer, normalized_sentence, return_unicode=False, sample=False)
                    else:
                        tokens = self.tokenizer.morphs(sentence)
                        tokens = post_processing(tokens)
                    #if int(label) > 0.5:
                    #    int_label = 1
                    #else:
                    #    int_label = 0
                    int_label = int(label)
                    data_set.append([tokens, int_label])
            with open(tokenized_corpus_fname, 'w') as f3:
                for tokens, label in data_set:
                    f3.writelines(' '.join(tokens) + "\u241E" + str(label) + "\n")
        return data_set, len(data_set)

    def train(self, sess, saver, global_step, output_feed):
        train_batches = self.get_batch(self.train_data, num_epochs=self.num_epochs, is_training=True)
        checkpoint_loss = 0.0
        train_batch_cnt = self.num_batches_per_epoch
        num_epochs = self.num_epochs
        train_cnt = 0
        train_num_data = 0
        n_epochs = 0
        for current_input_feed in train_batches:
            #_, _, _, current_loss = sess.run(output_feed, current_input_feed)
            _, _, current_logits, current_loss = sess.run(output_feed, current_input_feed)
            checkpoint_loss += current_loss
            current_preds = np.argmax(current_logits, axis=-1)
            sess_global_step = global_step.eval(sess)
            if(train_batch_count == 0):
                train_batch_cnt = self.num_batches_per_epoch
            if(train_batch_cnt > 0):
                n_epochs = int(train_cnt / train_batch_cnt)
            if(train_cnt % 100 == 0):
                #print(train_cnt, current_logits, ' preds : ', current_preds)
                print('train ({}) {} : {} / {}'.format(n_epochs, self.num_train_steps, train_batch_cnt, train_cnt))

            #if(train_cnt % 1000 == 0):
            if((global_step.eval(sess) % 1000) == 0):
            #if((global_step.eval(sess) % self.eval_every) == 0):
                tf.logging.info("global step %d train loss %.4f" %
                                (sess_global_step, checkpoint_loss / self.eval_every))
                checkpoint_loss = 0.0
                self.validation(sess, saver, global_step)
                #train_cnt = 0
            train_cnt += 1

        print('***** Eval results *****')
        print('average valid scores: {:.4f}'.format(np.average(self.valid_scores)))

    def validation(self, sess, saver, global_step):
        valid_loss, valid_pred, valid_num_data = 0, 0, 0
        valid_cnt = 0
        output_feed = [self.logits, self.loss]
        test_batches = self.get_batch(self.test_data, num_epochs=1, is_training=False)
        val_batch_cnt = self.num_batches_per_epoch

        for current_input_feed, current_labels in test_batches:
            current_logits, current_loss = sess.run(output_feed, current_input_feed)
            current_preds = np.argmax(current_logits, axis=-1)
            valid_loss += current_loss
            valid_num_data += len(current_labels)
            #print(current_logits)
            #print(current_preds)
            #tf.logging.info("current_logits.shape: " + str(np.array(current_logits).shape))
            #tf.logging.info("current_preds.shape: " + str(np.array(current_preds).shape) + ", current_labels.shape: " + str(np.array(current_labels).shape))

            for pred, label in zip(current_preds, current_labels):
                if pred == label:
                    valid_pred += 1
            
            if(valid_cnt % 100 == 0):
                #tf.logging.info("pred: " + str(pred) + ", label: " + str(label))
                #print("{} / {} : pred: {}, label: {}".format(valid_cnt, val_batch_cnt, str(pred), str(label)))
                #print("validation {} / {} : {} / {}".format(val_batch_cnt, valid_cnt, valid_pred, valid_num_data))
                #print("validation {} / {} : {} / {}".format(self.num_batches_per_epoch, valid_cnt, valid_pred, valid_num_data))
                print("validation {} / {}".format(self.num_batches_per_epoch, valid_cnt))
            valid_cnt += 1

        valid_score = valid_pred / valid_num_data
        tf.logging.info("valid loss %.4f valid score %.4f" %
                        (valid_loss, valid_score))
        if valid_score > self.best_valid_score:
            self.best_valid_score = valid_score
            path = self.model_save_path + "/" + str(valid_score)
            saver.save(sess, path, global_step=global_step)

        self.valid_scores.append(valid_score)

    def get_batch(self, data, num_epochs, is_training=True):
        if is_training:
            data_size = self.train_data_size
        else:
            data_size = self.test_data_size
        self.num_batches_per_epoch = int((data_size - 1) / self.batch_size)

        if is_training:
            tf.logging.info("num_batches_per_epoch : " + str(self.num_batches_per_epoch))
        for epoch in range(num_epochs):
            idx = random.sample(range(data_size), data_size)
            data = np.array(data)[idx]
            for batch_num in range(self.num_batches_per_epoch):
                batch_sentences = []
                batch_labels = []
                start_index = batch_num * self.batch_size
                end_index = (batch_num + 1) * self.batch_size
                features = data[start_index:end_index]
                for feature in features:
                    sentence, label = feature
                    batch_sentences.append(sentence)
                    batch_labels.append(int(label))
                yield self.make_input(batch_sentences, batch_labels, is_training)

    def make_input(self, sentences, labels, is_training):
        raise NotImplementedError

    def tune(self):
        raise NotImplementedError



class BERTTuner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname, vocab_fname,
                 pretrain_model_fname, bertconfig_fname, model_save_path,
                 max_seq_length=128, warmup_proportion=0.1,
                 batch_size=32, learning_rate=2e-5, num_labels=2):
        # Load a corpus.
        super().__init__(train_corpus_fname=train_corpus_fname,
                         tokenized_train_corpus_fname=train_corpus_fname + ".bert-tokenized",
                         test_corpus_fname=test_corpus_fname, batch_size=batch_size,
                         tokenized_test_corpus_fname=test_corpus_fname + ".bert-tokenized",
                         model_name="bert", vocab_fname=vocab_fname, model_save_path=model_save_path,
                         num_labels=num_labels)
        # configurations
        config = BertConfig.from_json_file(bertconfig_fname)
        self.pretrain_model_fname = pretrain_model_fname
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        #self.num_labels = 2 # positive, negative
        #self.num_labels = 59 # stt mcid class
        num_labels = self.num_labels
        self.PAD_INDEX = 0
        self.CLS_TOKEN = "[CLS]"
        self.SEP_TOKEN = "[SEP]"
        self.num_train_steps = (int((len(self.train_data) - 1) / self.batch_size) + 1) * self.num_epochs
        self.num_warmup_steps = int(self.num_train_steps * warmup_proportion)
        self.eval_every = int(self.num_train_steps / self.num_epochs)  # epoch마다 평가
        self.training = tf.placeholder(tf.bool)
        # build train graph
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids, self.logits, self.loss = make_bert_graph(config,
                                                                                                                    max_seq_length,
                                                                                                                    self.dropout_keep_prob_rate,
                                                                                                                    num_labels, tune=True)

    def tune(self):
        global_step = tf.train.get_or_create_global_step()
        tf.logging.info("num_train_steps: " + str(self.num_train_steps))
        tf.logging.info("num_warmup_steps: " + str(self.num_warmup_steps))
        train_op = create_optimizer(self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)
        output_feed = [train_op, global_step, self.logits, self.loss]
        restore_vars = [v for v in tf.trainable_variables() if "bert" in v.name]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(restore_vars).restore(sess, self.pretrain_model_fname)
        saver = tf.train.Saver(max_to_keep=1)
        self.train(sess, saver, global_step, output_feed)

    def make_input(self, sentences, labels, is_training):
        collated_batch = {'sequences': [], 'segments': [], 'masks': []}
        for tokens in sentences:
            tokens = tokens[:(self.max_seq_length - 2)]
            token_sequence = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
            segment = [0] * len(token_sequence)
            sequence = self.tokenizer.convert_tokens_to_ids(token_sequence)
            current_length = len(sequence)
            padding_length = self.max_seq_length - current_length
            collated_batch['sequences'].append(sequence + [self.PAD_INDEX] * padding_length)
            collated_batch['segments'].append(segment + [self.PAD_INDEX] * padding_length)
            collated_batch['masks'].append([1] * current_length + [self.PAD_INDEX] * padding_length)
        if is_training:
            input_feed = {
                self.training: is_training,
                self.input_ids: np.array(collated_batch['sequences']),
                self.segment_ids: np.array(collated_batch['segments']),
                self.input_mask: np.array(collated_batch['masks']),
                self.label_ids: np.array(labels)
            }
        else:
            input_feed_ = {
                self.training: is_training,
                self.input_ids: np.array(collated_batch['sequences']),
                self.segment_ids: np.array(collated_batch['segments']),
                self.input_mask: np.array(collated_batch['masks']),
                self.label_ids: np.array(labels)
            }
            input_feed = [input_feed_, labels]
        return input_feed

# deleted function WordEmbeddingTuner()

def allreduce_grads(all_grads, average):
    """
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.
    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.
    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        summed = nccl_ops.all_sum(grads)
        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower)
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)
    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables
    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]


def load_pretrained_xlnet_model(pretrained_model_fname, num_gpus):
    tf.logging.info("Initialize from the ckpt {}".format(pretrained_model_fname))
    name_to_variable = collections.OrderedDict()
    for var in tf.trainable_variables():
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(pretrained_model_fname)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        for k in range(num_gpus):
            assignment_map[name] = name_to_variable['tower' + str(k) + '/' + name]
    tf.train.init_from_checkpoint(pretrained_model_fname, assignment_map)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--train_corpus_fname', type=str, help='train corpus file name')
    parser.add_argument('--test_corpus_fname', type=str, help='test corpus file name')
    parser.add_argument('--vocab_fname', type=str, help='vocab file name')
    parser.add_argument('--pretrain_model_fname', type=str, help='pretrained model file name')
    parser.add_argument('--config_fname', type=str, help='config file name')
    parser.add_argument('--model_save_path', type=str, help='model save path')
    parser.add_argument('--embedding_name', type=str, help='embedding name')
    parser.add_argument('--embedding_fname', type=str, help='embedding file path')
    parser.add_argument('--num_gpus', type=str, help='number of GPUs (XLNet only)')
    parser.add_argument('--num_labels', type=int, help='number of labels)')
    args = parser.parse_args()

    if args.model_name == "bert":
        model = BERTTuner(train_corpus_fname=args.train_corpus_fname,
                          test_corpus_fname=args.test_corpus_fname,
                          vocab_fname=args.vocab_fname,
                          pretrain_model_fname=args.pretrain_model_fname,
                          bertconfig_fname=args.config_fname,
                          model_save_path=args.model_save_path,
                          num_labels=args.num_labels)
    else:
        model = None
    model.tune()
