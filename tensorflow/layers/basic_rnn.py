# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module provides wrappers for variants of RNN in Tensorflow
"""

import tensorflow as tf
import tensorflow.contrib as tc

#inputs:[max_p_num, max_p_len, embed_size]
def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
    Returns:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)

        outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
        print('rnn_outputs_shape',outputs.shape)
        print('rnn_states_shape', outputs.shape)
        #outputs:[max_p_num, max_p_len, hidden_size]
        #len(states) = num_layers
        if rnn_type.endswith('lstm'):
            c = [state.c for state in states]
            print('rnn_c_shape', outputs.shape)
            #c:[[max_p_num, hidden_size],....],每一层输出都有一个状态，类型一致。其中LSTM中每一层都以c, h
            h = [state.h for state in states] #c:[[max_p_num, hidden_size],....]
            print('rnn_h_shape', outputs.shape)
            states = h #states:[max_p_num, hidden_size]
            print('rnn_states01_shape', outputs.shape)
    else:
        cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
        )
        print('bi-LSTM_outputs_shape', outputs.shape)
        print('bi-LSTM_states_shape', outputs.shape)
        #outputs:[max_p_num, max_p_len, hidden_size]
        #states:[max_p_num, hidden_size]
        states_fw, states_bw = states
        if rnn_type.endswith('lstm'):
            c_fw = [state_fw.c for state_fw in states_fw]#:[max_p_num, hidden_size]
            h_fw = [state_fw.h for state_fw in states_fw]#:[max_p_num, hidden_size]
            print('bi-LSTM_h_fw_shape', outputs.shape)
            c_bw = [state_bw.c for state_bw in states_bw]#:[max_p_num, hidden_size]
            h_bw = [state_bw.h for state_bw in states_bw]#:[max_p_num, hidden_size]
            print('bi-LSTM_h_bw_shape', outputs.shape)
            states_fw, states_bw = h_fw, h_bw
        if concat:
            outputs = tf.concat(outputs, 2) #[max_p_len, max_p_num* hidden_size]
            print('concat_outputs_shape', outputs.shape)
            states = tf.concat([states_fw, states_bw], 1) #[max_p_num, 2 * hidden_size]
            print('concat_states_shape', outputs.shape)
        else:
            outputs = outputs[0] + outputs[1] #[2 * max_p_len, hidden_size]
            states = states_fw + states_bw #[2 * max_p_num, hidden_size]
    return outputs, states


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    cells = []
    for i in range(layer_num):
        if rnn_type.endswith('lstm'):
            cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        elif rnn_type.endswith('gru'):
            cell = tc.rnn.GRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell,
                                         input_keep_prob=dropout_keep_prob,
                                         output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells

