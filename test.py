import ipdb
# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data
from dnsc import DNSC
from dnsc2 import DNSC2
from dlstm import DLSTM
from mlstm import MLSTM
from mdhuapa import MDHUAPA
from mldhuapa import MLDHUAPA
from colored import fg, stylize
import dnsc
import dnsc2
import dlstm
import mlstm
import mdhuapa
import mldhuapa
import math
from tensorflow.python import debug as tf_debug


# delete all flags that remained by last run
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


try:
    del_all_flags(tf.flags.FLAGS)
except:
    pass


params = {
    'debug_params': [('debug', False, 'Whether to debug or not'),
                     ('check', False, 'Whether to make a checkpoint'),
                     ('checkpoint', '4725', 'checkpoint file'),
                     ('tfdbg', 'null', 'the manner to debug(cli or tensorboard)')],
    'data_params': [('cls_cnt', 10, "Numbers of class"),
                    ('dataset', 'test', "The dataset")],
    'model_chooing': [('model', 'dnsc', 'Model to train')],
    'model_hyperparam': [("emb_dim", 200, "Dimensionality of character embedding"),
                         ("hidden_size", 200, "hidden_size"),
                         ('max_sen_len', 50, 'max number of tokens per sentence'),
                         ('max_doc_len', 40, 'max number of tokens per sentence'),
                         ("lr", .001, "Learning rate"),
                         ("l2_rate", .001, "rate of l2 regularization"),
                         ("lambda1", .4, "rate of l2 regularization"),
                         ("lambda2", .3, "rate of l2 regularization"),
                         ("lambda3", .3, "rate of l2 regularization"),
                         ("bilstm", True, "use biLSTM or LSTM"),
                         ("sen_hop_cnt", 1, "number of hops in sentence layer"),
                         ("doc_hop_cnt", 1, "number of hops in document layer"),
                         ("convert_flag", 'obw', "params used in background converting process")],
    'training_params': [("batch_size", 1, "Batch Size"),
                        ("epoch_cnt", 1000, "Number of training epochs"),
                        ("evaluate_every", 200, "Evaluate model on dev set after this many steps"),
                        ("training_method", 'adam', 'Method chose to tune the weights')],
    'misc_params': [("allow_soft_placement", True, "Allow device soft device placement"),
                    ("log_device_placement", False, "Log placement of ops on devices")]
}

for param_collection in params.values():
    for param_name, default, description in param_collection:
        param_type = type(default)
        if param_type is int:
            tf.flags.DEFINE_integer(param_name, default, description)
        elif param_type is float:
            tf.flags.DEFINE_float(param_name, default, description)
        elif param_type is str:
            tf.flags.DEFINE_string(param_name, default, description)
        elif param_type is bool:
            tf.flags.DEFINE_boolean(param_name, default, description)

flags = tf.flags.FLAGS

# save current codes
cur_time = time.time()
os.system('mkdir code_history/' + str(cur_time))
os.system('cp *.py code_history/' + str(cur_time) + '/')
localtime = time.localtime(cur_time)

_ = flags.batch_size
# print params
print("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")


graph = tf.Graph()
with graph.as_default():
    # Load data
    print("Loading data...")
    datasets = ['data/' + flags.dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']]
    embeddingpath = 'data/' + flags.dataset + '/embedding.txt'
    hierarchy = flags.model in ['dnsc']
    datasets, lengths, embedding, usr_cnt, prd_cnt, wrd_dict = \
        data.build_dataset(datasets, embeddingpath, flags.max_doc_len,
                           flags.max_sen_len, hierarchy)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    trainset = trainset.batch(flags.batch_size)
    devset = devset.batch(flags.batch_size)
    testset = testset.batch(flags.batch_size)
    print("Loading data finished...")

    # create the session
    session_config = tf.ConfigProto(
        allow_soft_placement=flags.allow_soft_placement,
        log_device_placement=flags.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    if flags.tfdbg == 'cli':
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess.as_default():
        # build the model
        model_params = {
            'max_sen_len': flags.max_sen_len, 'max_doc_len': flags.max_doc_len,
            'cls_cnt': flags.cls_cnt, 'embedding': embedding,
            'emb_dim': flags.emb_dim, 'hidden_size': flags.hidden_size,
            'usr_cnt': usr_cnt, 'prd_cnt': prd_cnt,
            'l2_rate': flags.l2_rate, 'sen_hop_cnt': flags.sen_hop_cnt,
            'doc_hop_cnt': flags.doc_hop_cnt,
            'debug': flags.debug, 'convert_flag': flags.convert_flag,
            'lambda1': flags.lambda1, 'lambda2': flags.lambda2, 'lambda3': flags.lambda3
        }
        if flags.model == 'dnsc':
            model = DNSC(model_params)
            cur_model = dnsc
        elif flags.model == 'dnsc2':
            model = DNSC2(model_params)
            cur_model = dnsc2
        if flags.model == 'dlstm':
            model = DLSTM(model_params)
            cur_model = dlstm
        if flags.model == 'mlstm':
            model = MLSTM(model_params)
            cur_model = mlstm
        if flags.model == 'mdhuapa':
            model = MDHUAPA(model_params)
            cur_model = mdhuapa
        if flags.model == 'mldhuapa':
            model = MLDHUAPA(model_params)
            cur_model = mldhuapa

        data_iter = tf.data.Iterator.from_structure(trainset.output_types,
                                                    output_shapes=trainset.output_shapes)
        traininit = data_iter.make_initializer(trainset)
        devinit = data_iter.make_initializer(devset)
        testinit = data_iter.make_initializer(testset)

        metrics = model.build(data_iter)

        # restore the params
        saver = tf.train.Saver()
        saver.restore(sess, 'check_points/-' + flags.checkpoint)

        wrd = 'divide0/fold/cur_wrd:0'
        usr_attention = []
        prd_attention = []
        for i in range(flags.sen_hop_cnt):
            usr_attention.append('divide0/user_block/sentence_layer/hop' + str(i) +
                                 '/attention/attention_without_null_word:0')
            prd_attention.append('divide0/product_block/sentence_layer/hop' + str(i) +
                                 '/attention/attention_without_null_word:0')
        usr_doc_att = 'divide0/user_block/document_layer/hop0/attention/attention_without_null_word:0'
        prd_doc_att = 'divide0/product_block/document_layer/hop0/attention/attention_without_null_word:0'

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # run a dataset

        def run_set(sess, testlen, metrics, ops=tuple()):
            ops = tuple(ops)
            global flags
            metrics_total = [0] * len(metrics)
            op_results = [[] for i in ops]
            ipdb.set_trace()
            for i in range(int(math.ceil(float(testlen) / flags.batch_size))):
                cur_metrics = sess.run(metrics + ops)
                if flags.tfdbg == 'null':
                    sen_hop_cnt = flags.sen_hop_cnt
                    cur_usr_doc_att = np.transpose(cur_metrics[-2 * sen_hop_cnt - 3], [2, 1, 0])
                    cur_prd_doc_att = np.transpose(cur_metrics[-2 * sen_hop_cnt - 2], [2, 1, 0])
                    cur_wrd = cur_metrics[-2 * sen_hop_cnt - 1].flatten()
                    cur_wrd = [wrd_dict[i] for i in cur_wrd]
                    for i, color in zip(range(sen_hop_cnt), ['red', 'green', 'yellow']):
                        cur_usr_att = (cur_usr_doc_att * cur_metrics[-2 * sen_hop_cnt + i]).flatten()
                        cur_prd_att = (cur_prd_doc_att * cur_metrics[-1 * sen_hop_cnt + i]).flatten()
                        print fg(color), cur_wrd[np.argmax(cur_usr_att)], np.max(cur_usr_att), np.argmax(cur_usr_att), '\t',\
                            cur_wrd[np.argmax(cur_prd_att)], np.max(cur_prd_att), np.argmax(cur_prd_att)
                    for i in range(4 - sen_hop_cnt):
                        print('-' * 50)
                    # cur_usr_att0 = (cur_usr_doc_att * cur_metrics[-4]).flatten()
                    # cur_usr_att1 = (cur_usr_doc_att * cur_metrics[-3]).flatten()
                    # cur_prd_att0 = (cur_prd_doc_att * cur_metrics[-2]).flatten()
                    # cur_prd_att1 = (cur_prd_doc_att * cur_metrics[-1]).flatten()
                    # print fg('green'), cur_wrd[np.argmax(cur_usr_att1)], np.max(cur_usr_att1), np.argmax(cur_usr_att1), '\t',\
                    #     cur_wrd[np.argmax(cur_prd_att1)], np.max(cur_prd_att1), np.argmax(cur_prd_att1)
                for j in range(len(metrics)):
                    metrics_total[j] += cur_metrics[j]
                for j in range(len(ops)):
                    op_results[j].append(cur_metrics[len(metrics) + j])
            return [metrics_total] + op_results

        try:
            for epoch in range(flags.epoch_cnt):
                sess.run(traininit)
                output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
                # train on trainset
                ops = [usr_doc_att, prd_doc_att, wrd]
                ops += usr_attention
                ops += prd_attention
                train_metrics, step, cur_wrd, cur_attention0, cur_attention1 = \
                    run_set(sess, trainlen, metrics, ops)
                info = model.output_metrics(train_metrics, trainlen)
                info = 'Trainset:' + info
                print(stylize(info, fg('yellow')))
                print >> output_file, info

                # test on devset
                sess.run(devinit)
                dev_metrics, = run_set(sess, devlen, metrics)
                info = model.output_metrics(dev_metrics, devlen)
                info = 'Devset:  ' + info
                print(stylize(info, fg('green')))
                print >> output_file, info

                # test on testset
                sess.run(testinit)
                test_metrics, = run_set(sess, testlen, metrics)
                info = model.output_metrics(test_metrics, testlen)
                info = 'Testset: ' + info
                print(stylize(info, fg('red')))
                print >> output_file, info

                # print info of this epoch
                info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
                info = 'Epoch %d finished, ' % epoch + info
                print(stylize(info, fg('white')))
                print >> output_file, info

        except KeyboardInterrupt:
            print('Interrupted')
            best_test_acc = model.best_test_acc
            src = 'code_history/' + str(cur_time)
            dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
            os.system('mv ' + src + ' ' + dest)
