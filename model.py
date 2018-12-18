import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
from math import sqrt
import numpy as np
import capslayer as cl
from utils import lstm
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class CSC(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.B = args['hidden_size']
        self.C = args['hidden_size']
        self.D = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        # self.convert_flag = args['convert_flag']
        # self.debug = args['debug']
        # self.lambda1 = args['lambda1']
        # self.lambda2 = args['lambda2']
        # self.lambda3 = args['lambda3']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.initializers.zeros()
        self.emb_initializer = tf.contrib.layers.xavier_initializer()
        # self.weights_initializer = tf.initializers.random_uniform(-.01, .01)
        # self.biases_initializer = tf.initializers.random_uniform(-.01, .01)
        # self.emb_initializer = tf.initializers.random_uniform(-.01, .01)

        hsize = self.hidden_size

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb': const(self.embedding, name='wrd_emb', dtype=tf.float32),
                # 'wrd_emb': tf.Variable(embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], self.emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], self.emb_initializer)
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def get_weight(self, name, shape):
        return var(name, shape, self.weights_initializer)

    def get_bias(self, name, shape):
        return var(name, shape, self.biases_initializer)

    def csc(self, x, max_sen_len, max_doc_len, sen_len, doc_len, usr, prd, convert_flag):
        # spilt each batch into max_doc_len ones
        sen_x = tf.reshape(x, [-1, 1, max_sen_len, self.emb_dim], name='sentence_x')
        sen_len = tf.reshape(sen_len, [-1], name='sen_len')

        B, C, D, E, K1, K2, d = self.B, self.C, self.D, self.E, 5, 3, 16
        # conv1 should be a tensor with shape
        # [batch_size*max_doc_len, max_sen_len-K1+1, 1, B]
        # here a real batch is considered as max_doc_len batches
        conv1 = tf.layers.conv2d(inputs=sen_x, filters=B,
                                 kernel_size=[K1, self.emb_dim],
                                 name='embedding -> conv1')
        conv1 = tf.reshape(conv1, [-1, max_sen_len - K1 + 1, B, 1], name='conv1')

        # primaryCaps should be a tensor with shape
        # [batch_size*max_doc_len, max_sen_len-K1+1, 1, C, d, 1]
        # here a real batch is considered as max_doc_len batches
        primaryCaps, _activation = \
            cl.layers.primaryCaps(conv1, filters=C, kernel_size=[1, B],
                                  strides=1, out_caps_dims=[d, 1],
                                  name='conv1 -> primaryCaps')
        primaryCaps = tf.reshape(primaryCaps, [-1, max_sen_len - K1 + 1, C, 1, d, 1],
                                 name='primaryCaps')

        # convCaps should be a tensor with shape
        # [batch_size*max_doc_len, max_sen_len-K1-K2+2, 1, D, d, 1]
        # here a real batch is considered as max_doc_len batches
        convCaps, _activation = cl.layers.conv2d(inputs=primaryCaps, activation=None,
                                                 filters=D, out_caps_dims=[d, 1],
                                                 kernel_size=[K2, C], strides=[1, 1],
                                                 routing_method='DynamicRouting',
                                                 name='primaryCaps -> convCaps')
        convCaps = tf.reshape(convCaps, [-1, max_sen_len - K1 - K2 + 2, D, 1, d, 1],
                              name='convCaps')
        flatten = tf.reshape(convCaps, [-1, (max_sen_len - K1 - K2 + 2) * D, d, 1],
                             name='flatten')

        # FCCaps should be a tensor with shape
        # [batch_size*max_doc_len, E, d, 1]
        # here a real batch is considered as max_doc_len batches
        FCCaps = cl.layers.dense(inputs=flatten, activation=None, num_outputs=E,
                                 out_caps_dims=[d, 1], routing_method='DynamicRouting',
                                 name='flatten -> FCCaps')
        FCCaps = tf.reshape(FCCaps, [-1, E, d], name='FCCaps')
        aspects_sentences = tf.reshape(FCCaps, [-1, max_doc_len, E, d], name='aspects_sentences')

        # aspects_document should be a tensor with shape
        # [batch_size, E, d]
        aspects_document = tf.reduce_sum(aspects_sentences, axis=1, name='aspects_document')
        flatten_aspects = tf.reduce_sum(aspects_document ** 2, axis=2)
        # flatten_aspects should be a tensor with shape
        # [batch_size, E]
        flatten_aspects = tf.sqrt(flatten_aspects, name='flatten_aspects')

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, sen_len, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'])

            usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            input_x = lookup(self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')

        # build the process of model
        d_hat, d_hatu, d_hatp = self.dhuapa(input_x, self.max_sen_len, self.max_doc_len,
                                            sen_len, doc_len, usr, prd, self.convert_flag)
        prediction = tf.argmax(d_hat, 1, name='prediction')
        # predictionu = tf.argmax(d_hatu, 1, name='predictionu')
        # predictionp = tf.argmax(d_hatp, 1, name='predictionp')

        with tf.variable_scope("loss"):
            # !!!
            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            self.loss = sce(logits=d_hat, labels=tf.one_hot(input_y, self.cls_cnt))
            lossu = sce(logits=d_hatu, labels=tf.one_hot(input_y, self.cls_cnt))
            lossp = sce(logits=d_hatp, labels=tf.one_hot(input_y, self.cls_cnt))

            self.loss = self.lambda1 * self.loss + self.lambda2 * lossu + self.lambda3 * lossp
            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            self.loss = tf.reduce_mean(self.loss) + self.l2_rate * regularizer

        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return self.loss, mse, correct_num, accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, accuracy = metrics
        info = 'Loss = %.3f, RMSE = %.3f, Acc = %.3f' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        _test_loss, test_mse, test_correct_num, test_accuracy = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        test_rmse = sqrt(float(test_mse) / testlen)
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            self.best_test_rmse = test_rmse
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
