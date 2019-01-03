import pandas as pd
import tensorflow as tf
from functools import partial
import numpy as np
import os
import csv


reading_col_name = ['usr', 'prd', 'rating', 'content']
output_col_name = ['usr', 'prd', 'rating', 'content', 'doc_len', 'sen_len']


def build_dataset(filenames, tfrecords_filenames, stats_filename, embedding_filename,
                  max_doc_len, max_sen_len, hierarchy, emb_dim):
    datasets = []
    wrd_dict, wrd_index, embedding = load_embedding(embedding_filename, emb_dim)

    tfrecords_filenames = [i + str(hierarchy) + str(max_doc_len) + str(max_sen_len) for i in tfrecords_filenames]
    stats = {}
    if sum([os.path.exists(i) for i in tfrecords_filenames]) < len(tfrecords_filenames) \
            or not os.path.exists(stats_filename):
        for tfrecords_filename in tfrecords_filenames:
            if os.path.exists(tfrecords_filename):
                os.remove(tfrecords_filename)
        if os.path.exists(stats_filename):
            os.remove(stats_filename)
        # read the data and transform them
        data_frames, stats['usr_cnt'], stats['prd_cnt'] = \
            read_files(filenames, wrd_index, max_doc_len, max_sen_len, hierarchy)

        # build the dataset
        for filename, tfrecords_filename, data_frame in zip(filenames, tfrecords_filenames, data_frames):
            data_frame['content'] = data_frame['content'].transform(lambda x: x.tostring())
            data_frame['sen_len'] = data_frame['sen_len'].transform(lambda x: x.tostring())

            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            for item in data_frame.iterrows():
                def int64list(value):
                    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

                def byteslist(value):
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

                feature = {'usr': int64list([item[1]['usr']]),
                           'prd': int64list([item[1]['prd']]),
                           'rating': int64list([item[1]['rating']]),
                           'content': byteslist([item[1]['content']])}
                feature['sen_len'] = byteslist([item[1]['sen_len']])
                feature['doc_len'] = int64list([item[1]['doc_len']])

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            stats[filename + 'len'] = len(data_frame)
            # lengths.append(len(data_frame))

        stats_file = csv.writer(open(stats_filename, 'w'))
        # print('usr_cnt: %d, prd_cnt: %d' % (usr_cnt, prd_cnt))
        for key, val in stats.items():
            stats_file.writerow([key, val])

    def transform_example(example):
        dics = {
            'usr': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'prd': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'rating': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'content': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)}
        dics['sen_len'] = tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)
        dics['doc_len'] = tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)

        ans = tf.parse_single_example(example, dics)
        ans['content'] = tf.decode_raw(ans['content'], tf.int64)
        ans['sen_len'] = tf.decode_raw(ans['sen_len'], tf.int64)
        return ans

    for key, val in csv.reader(open(stats_filename)):
        stats[key] = int(val)
    for tfrecords_filename in tfrecords_filenames:
        dataset = tf.data.TFRecordDataset(tfrecords_filename)
        dataset = dataset.map(transform_example)
        datasets.append(dataset)

    lengths = [stats[filename + 'len'] for filename in filenames]
    return datasets, lengths, embedding.values, stats['usr_cnt'], stats['prd_cnt'], wrd_dict


# load an embedding file
def load_embedding(filename, emb_dim):
    emb_col_name = ['wrd'] + [i for i in range(emb_dim)]
    data_frame = pd.read_csv(filename, sep=' ', header=0, names=emb_col_name)
    embedding = data_frame[emb_col_name[1:]]
    wrd_dict = data_frame['wrd'].tolist()
    wrd_index = {s: i for i, s in enumerate(wrd_dict)}
    return wrd_dict, wrd_index, embedding


# transform a sentence into indices
def sentence_transform(document, wrd_index, max_doc_len, max_sen_len, hierarchy):
    if hierarchy:
        sentence_index = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        for i, sentence in enumerate(document):
            if i >= max_doc_len:
                break
            j = 0
            for wrd in sentence:
                if j >= max_sen_len:
                    break
                if wrd in wrd_index:
                    sentence_index[i][j] = wrd_index[wrd]
                    j += 1
    else:
        sentence_index = np.zeros((max_doc_len * max_sen_len, ), dtype=np.int)
        i = 0
        for wrd in document:
            if i == max_doc_len * max_sen_len:
                break
            if wrd in wrd_index:
                sentence_index[i] = wrd_index[wrd]
                i += 1
        sentence_index = sentence_index.reshape((max_doc_len, max_sen_len))
    return sentence_index


def split_paragraph(paragraph, hierarchy):
    if hierarchy:
        sentences = paragraph.split('<sssss>')
        for i, _ in enumerate(sentences):
            sentences[i] = sentences[i].split()
    else:
        sentences = paragraph.split()
    return sentences


def read_files(filenames, wrd_index, max_doc_len, max_sen_len, hierarchy):
    data_frames = [pd.read_csv(filename, sep='\t\t', names=reading_col_name, engine='python')
                   for filename in filenames]
    print('Data frame loaded.')

    # count contents' length
    for df in data_frames:
        df['content'] = df['content'].transform(partial(split_paragraph, hierarchy=hierarchy))
        df['rating'] = df['rating'] - 1
        # df['max_sen_len'] = df['sen_len'].transform(lambda sen_len: max(sen_len))

    total_data = pd.concat(data_frames)
    # max_doc_len = total_data['doc_len'].max()
    # max_sen_len = total_data['max_sen_len'].max()
    print('Length counted')

    # transform users and products to indices
    usr = total_data['usr'].unique().tolist()
    usr = {name: index for index, name in enumerate(usr)}
    prd = total_data['prd'].unique().tolist()
    prd = {name: index for index, name in enumerate(prd)}
    for df in data_frames:
        df['usr'] = df['usr'].map(usr)
        df['prd'] = df['prd'].map(prd)
    print('Users and products indexed.')

    # transform contents into indices
    for df in data_frames:
        df['content'] = df['content'].transform(
            partial(sentence_transform, wrd_index=wrd_index, max_doc_len=max_doc_len,
                    max_sen_len=max_sen_len, hierarchy=hierarchy))
        df['sen_len'] = df['content'].transform(lambda i: np.count_nonzero(i, axis=1))
        df['doc_len'] = df['sen_len'].transform(lambda i: np.count_nonzero(i))
    print('Contents indexed.')

    return data_frames, len(usr), len(prd)
