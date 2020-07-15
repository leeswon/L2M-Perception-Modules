import numpy as np
import tensorflow as tf

_tf_ver = tf.__version__.split('.')
if int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14):
    from tensorflow.compat.v1 import trainable_variables
    _tf_tensor = tf.is_tensor
else:
    from tensorflow import trainable_variables
    _tf_tensor = tf.contrib.framework.is_tensor

#### Miscellaneous functions

def get_list_of_valid_tensors(list_of_variables):
    list_of_valid_tensors = []
    for elem in list_of_variables:
        #if elem is not None:
        if elem in tf.global_variables():
            list_of_valid_tensors.append(elem)
    return list_of_valid_tensors

def get_value_of_valid_tensors(tf_sess, list_of_variables):
    list_of_val = []
    for elem in list_of_variables:
        list_of_val.append(elem if (elem is None) else tf_sess.run(elem))
    return list_of_val

def count_trainable_var(list_params):
    total_para_cnt = 0
    for var in list_params:
        para_cnt_tmp = 1
        if type(var) == np.ndarray:
            for dim in var.shape:
                para_cnt_tmp *= int(dim)
        elif _tf_tensor(var):
            for dim in var.get_shape():
                para_cnt_tmp *= int(dim)
        else:
            para_cnt_tmp = 0
        total_para_cnt += para_cnt_tmp
    return total_para_cnt


#### functions to build neural networks

## function to generate weight parameter
def new_weight(shape, trainable=True, init_tensor=None, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05) if init_tensor is None else init_tensor, trainable=trainable, name=name)

## function to generate bias parameter
def new_bias(shape, trainable=True, init_val=0.2, init_tensor=None, name=None):
    return tf.Variable(tf.constant(init_val, dtype=tf.float32, shape=shape) if init_tensor is None else init_tensor, trainable=trainable, name=name)

def new_DFCNN_KB_param(shape, layer_number, task_number, reg_type, init_tensor=None, trainable=True):
    #kb_name = 'KB_'+str(layer_number)+'_'+str(task_number)
    kb_name = 'KB_'+str(layer_number)
    if init_tensor is None:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, trainable=trainable)
    elif type(init_tensor) == np.ndarray:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, initializer=tf.constant_initializer(init_tensor), trainable=trainable)
    else:
        param_to_return = init_tensor
    return param_to_return

## function to generate task-specific parameters for ELLA_tensorfactor layer
def new_DFCNN_TS_param(shape, layer_number, task_number, reg_type, init_tensor, trainable):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_ConvW1_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    params_to_return, params_name = [], [ts_w_name, ts_b_name, ts_k_name, ts_p_name]
    for i, (t, n) in enumerate(zip(init_tensor, params_name)):
        if t is None:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable))
        elif type(t) == np.ndarray:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable, initializer=tf.constant_initializer(t)))
        else:
            params_to_return.append(t)
    return params_to_return





## function to add fully-connected layer
def new_fc_layer(layer_input, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None, trainable=True, use_numpy_var_in_graph=False):
    input_dim = int(layer_input.shape[1])
    with tf.name_scope('fc_layer'):
        if weight is None:
            weight = new_weight(shape=[input_dim, output_dim], trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=[input_dim, output_dim], init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[output_dim], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[output_dim], init_tensor=bias, trainable=trainable)

        if activation_fn is None:
            layer = tf.matmul(layer_input, weight) + bias
        elif activation_fn is 'classification':
            layer = tf.matmul(layer_input, weight) + bias
        else:
            layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None, tensorboard_name_scope='fc_net', trainable=True, use_numpy_var_in_graph=False):
    if params is None:
        params = [None for _ in range(2*len(dim_layers))]

    layers, params_to_return = [], []
    if len(dim_layers) < 1:
        #### for the case that hard-parameter shared network does not have shared layers
        layers.append(net_input)
    else:
        with tf.name_scope(tensorboard_name_scope):
            for cnt in range(len(dim_layers)):
                if cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(net_input, dim_layers[cnt], activation_fn=activation_fn, weight=params[0], bias=params[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is 'classification':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt],  activation_fn='classification', weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is None:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is 'same':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                else:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                layers.append(layer_tmp)
                params_to_return = params_to_return + para_tmp
    return (layers, params_to_return)

#### function to add 2D convolutional layer
def new_conv_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, use_numpy_var_in_graph=False, name_scope='conv_layer'):
    with tf.name_scope(name_scope):
        if weight is None:
            weight = new_weight(shape=k_size, trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=k_size, init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[k_size[-1]], init_tensor=bias, trainable=trainable)

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type) + bias

        if not (activation_fn is None):
            conv_layer = activation_fn(conv_layer)

        if max_pooling and (pool_size[1] > 1 or pool_size[2] > 1):
            layer = tf.nn.max_pool(conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = conv_layer
    return (layer, [weight, bias])





def new_hybrid_DFCNN_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, trainable=True, trainable_KB=True):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        ## KB \in R^{1 \times h \times w \times c}
        KB_param = new_DFCNN_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type, KB_param, trainable=trainable_KB)

        ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
        ## TS2 : Deconv bias \in R^{kb_c_out}
        ## TS3 : tensor W \in R^{kb_c_out \times ch_in \times ch_out}
        ## TS4 : Conv bias \in R^{ch_out}
        TS_param = new_DFCNN_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[1]], [1, 1, 1, TS_size[1]], [TS_size[1], ch_size[0], ch_size[1]], [ch_size[1]]], layer_num, task_num, TS_reg_type, [None, None, None, None] if TS_param is None else TS_param, trainable=trainable)

    with tf.name_scope('DFCNN_param_gen'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1]])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        W = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
        b = TS_param[3]

    layer_eqn, _ = new_conv_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param, [W, b]



def new_hybrid_DFCNN_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, task_index=0, trainable=True, trainable_KB=True):
    _num_TS_param_per_layer = 4

    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing), len(cnn_KB_sizes)//2, len(cnn_TS_sizes)//2, len(cnn_TS_stride_sizes)//2]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]

    ## add CNN layers
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(cnn_KB_params is None and cnn_TS_params is None), (not (cnn_KB_params is None) and (cnn_TS_params is None)), not (cnn_KB_params is None or cnn_TS_params is None), ((cnn_KB_params is None) and not (cnn_TS_params is None))]
    if control_flag[1]:
        cnn_TS_params = []
    elif control_flag[3]:
        cnn_KB_params = []
    elif control_flag[0]:
        cnn_KB_params, cnn_TS_params = [], []
    cnn_gen_params = []

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]

    with tf.name_scope('Hybrid_DFCNN'):
        cnn_model, cnn_params_to_return = [], []
        cnn_KB_to_return, cnn_TS_to_return = [], []
        for layer_cnt in range(num_conv_layers):
            KB_para_tmp, TS_para_tmp, para_tmp = [None], [None for _ in range(_num_TS_param_per_layer)], [None, None]
            cnn_gen_para_tmp = [None, None]

            if layer_cnt == 0:
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_conv_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable)
            else:
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_hybrid_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_conv_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable)

            cnn_model.append(layer_tmp)
            cnn_KB_to_return = cnn_KB_to_return + KB_para_tmp
            cnn_TS_to_return = cnn_TS_to_return + TS_para_tmp
            cnn_params_to_return = cnn_params_to_return + para_tmp
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1]*cnn_model[-1].shape[2]*cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    return (cnn_model+fc_model, cnn_KB_to_return, cnn_TS_to_return, cnn_gen_params, cnn_params_to_return, fc_params)
