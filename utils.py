import tensorflow as tf


def lstm(inputs, sequence_length, hidden_size, scope):
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, initializer=tf.contrib.layers.xavier_initializer())
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, initializer=tf.contrib.layers.xavier_initializer())
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
    outputs = tf.concat(outputs, axis=2)
    return outputs, state

def get_shape(tensor):
    pass
