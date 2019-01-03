# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import capslayer as cl


def transforming(inputs, num_outputs, out_caps_dims, share, transform,
                 identity=None, identity_dim=None, name=None):
    """
    Args:
        inputs: A 4-D or 6-D tensor, [batch_size, num_inputs] + in_caps_dims or [batch_size, height, width, channels] + in_caps_dims.
        num_outputs: Integer, the number of output capsules.
        out_caps_dims: A list of 2 integers. The dimensions of output capsule, e.g. out_caps_dims=[4, 4].
        name: String, a name for this operation.

    Returns:
        votes: A 5-D or 7-D tensor, [batch_size, num_inputs, num_outputs] + out_caps_dims or [batch_size, height, width, channels, num_outputs] + out_caps_dims.
    """
    name = "transforming" if name is None else name
    with tf.variable_scope(name) as scope:
        input_shape = cl.shape(inputs)
        prefix_shape = [1 for i in range(len(input_shape) - 3)] + input_shape[-3:-2] + [num_outputs]
        prefix_shape[1] = 1
        in_caps_dims = input_shape[-2:]

        # if share is True:
        #     shape = prefix_shape + [1, out_caps_dims[0], 1]
        # else:
        shape = prefix_shape + [in_caps_dims[0], out_caps_dims[0], 1]
        expand_axis = -2
        reduce_sum_axis = -3

        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=expand_axis)
        transform_mat = tf.get_variable("transformation_matrix", shape=shape)
        bias = tf.get_variable('transformation_bias', shape=[num_outputs] + out_caps_dims)
        if transform:
            votes = tf.reduce_sum(in_pose * transform_mat, axis=reduce_sum_axis)
            votes += bias
        else:
            votes = in_pose + bias[:, :, :, None]
            votes = tf.reduce_sum(in_pose, axis=-1)

        if identity is not None:
            dim = shape[-2]
            for n, i in enumerate(identity):
                bias_mat = tf.get_variable('transformation_bias' + str(n), shape=[identity_dim, dim])
                bias = tf.matmul(i, bias_mat)[:, None, None, :, None]
                votes += bias

        return votes
