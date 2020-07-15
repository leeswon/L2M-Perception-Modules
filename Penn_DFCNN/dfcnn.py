import numpy as np
import tensorflow as tf
from perceptionAPI import L2MClassifier
from Penn_DFCNN.utils_dfcnn import count_trainable_var, get_list_of_valid_tensors, new_hybrid_DFCNN_net

_tf_ver = tf.__version__.split('.')
if int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14):
    from tensorflow.compat.v1 import trainable_variables
    _tf_tensor = tf.is_tensor
else:
    from tensorflow import trainable_variables
    _tf_tensor = tf.contrib.framework.is_tensor

class DeconvolutionalFactorizedCNN(L2MClassifier):
    def __init__(self, model_hyperparam):
        #super.__init__(model_hyperparam)

        # Hyper-parameters to initialize DF-CNN
        self.input_size = model_hyperparam['image_dimension']    ## img_width * img_height * img_channel
        self.cnn_channels_size = [self.input_size[-1]]+list(model_hyperparam['channel_sizes'])    ## include dim of input channel
        self.cnn_kernel_size = model_hyperparam['kernel_sizes']     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = model_hyperparam['stride_sizes']
        self.pool_size = model_hyperparam['pooling_size']      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = model_hyperparam['dense_layer']
        self.padding_type = model_hyperparam['padding_type']
        self.max_pooling = model_hyperparam['max_pooling']
        self.dropout = model_hyperparam['dropout']

        self.conv_transfer_config = model_hyperparam['conv_transfer_config']
        self.dfcnn_KB_size = model_hyperparam['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperparam['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperparam['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperparam['regularization_scale'][0]
        self.dfcnn_TS_reg_scale = model_hyperparam['regularization_scale'][1]

        self.learn_rate = model_hyperparam['lr']
        # self.batch_size = model_hyperparam['batch_size']
        self.hidden_act = model_hyperparam['hidden_activation']

        self.num_conv_layers, self.num_fc_layers = len(self.cnn_channels_size)-1, len(self.fc_size)+1

        self.num_tasks = 0
        self.num_trainable_var, self.dfcnn_KB_params_size = 0, 0
        self.output_sizes = []
        self.task_indices = []

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.Session()

        #### placeholder of model
        self.dropout_prob = tf.placeholder(dtype=tf.float32)
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size[0], self.input_size[1], self.input_size[2]])
        self.x_batch, self.y_batch = [], []

        #### tensorflow graph (model) and variables (parameters)
        self.task_models  = []
        self.conv_params, self.fc_params, self.params = [], [], []
        self.dfcnn_KB_params, self.dfcnn_TS_params, self.dfcnn_gen_conv_params = None, [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)

        self.eval, self.pred = [], []
        self.loss, self.accuracy = [], []
        self.update = []

    def is_new_task(self, curr_task_index):
        return (not (curr_task_index in self.task_indices))

    def find_task_model(self, task_index_to_search):
        return self.task_indices.index(task_index_to_search)

    def number_of_learned_tasks(self):
        return self.num_tasks

    def _build_task_model(self, net_input, output_size, task_cnt, params):
        params_KB, params_TS, params_conv, params_fc = params['KB'], params['TS'], params['Conv'], params['FC']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, fc_params = new_hybrid_DFCNN_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_transfer_config, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, trainable=True)
        if self.dfcnn_KB_params is None:
            self.dfcnn_KB_params = dfcnn_KB_param_tmp
        return task_net, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, fc_params

    def addNewTask(self, task_info, num_classes):
        # Generate/initialize task-specific sub-modules
        # task_info contains 'task_index' (enumeration of tasks) and 'task_description' (details of task)
        # num_classes is for the output size of task-specific sub-module

        self.num_tasks += 1
        self.output_sizes.append(num_classes)
        self.task_indices.append(task_info['task_index'])

        #### placeholder of model
        self.x_batch.append(self.model_input)
        self.y_batch.append(tf.placeholder(dtype=tf.float32, shape=[None]))

        #### generate new task model
        task_net, dfcnn_TS_params, gen_conv_params, conv_params, fc_params = self._build_task_model(self.x_batch[-1], num_classes, self.num_tasks-1, params={'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None})

        #### Collect parameters of task models
        self.task_models.append(task_net)
        self.dfcnn_TS_params.append(dfcnn_TS_params)
        self.dfcnn_gen_conv_params.append(gen_conv_params)
        self.conv_params.append(conv_params)
        self.fc_params.append(fc_params)
        self.params.append(self._collect_variables())

        if self.num_tasks == 1:
            self.dfcnn_KB_params_size = count_trainable_var(self.dfcnn_KB_params)
            self.num_trainable_var += count_trainable_var(self.params[-1])
        else:
            self.num_trainable_var += count_trainable_var(self.params[-1]) - self.dfcnn_KB_params_size

        self.update_eval()
        self.update_loss()
        self.update_accuracy()
        self.update_opt()

    def update_eval(self):
        self.eval.append(tf.nn.softmax(self.task_models[-1][-1]))
        self.pred.append(tf.argmax(self.task_models[-1][-1], 1))

    def update_loss(self):
        self.loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.y_batch[-1], tf.int32), logits=self.task_models[-1][-1])))

    def update_accuracy(self):
        self.accuracy.append(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.task_models[-1][-1], 1), tf.cast(self.y_batch[-1], tf.int64)), tf.float32)))

    def update_opt(self):
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
        TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

        #### Collect only trainable parameters (parameters of current task model)
        dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[-1])
        conv_trainable_param = get_list_of_valid_tensors(self.conv_params[-1])
        fc_trainable_param = get_list_of_valid_tensors(self.fc_params[-1])

        KB_grads = tf.gradients(self.loss[-1] + KB_reg_term2, dfcnn_KB_trainable_param)
        KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, dfcnn_KB_trainable_param)]

        TS_grads = tf.gradients(self.loss[-1] + TS_reg_term2, dfcnn_TS_trainable_param)
        TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, dfcnn_TS_trainable_param)]

        conv_grads = tf.gradients(self.loss[-1], conv_trainable_param)
        conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, conv_trainable_param)]

        fc_grads = tf.gradients(self.loss[-1], fc_trainable_param)
        fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, fc_trainable_param)]

        grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate)
        self.update.append(trainer.apply_gradients(grads_vars))

    def inference(self, task_info, X):
        # Make inference on the given data X according to the task (task_info)
        # return y

        task_model_index = self.find_task_model(task_info['task_index'])
        pred_y = self.sess.run(self.pred[task_model_index], feed_dict={self.model_input: X, self.dropout_prob: 1.0})
        return pred_y

    def train(self, task_info, X, y):
        # Optimize trainable parameters according to the task (task_info) and data (X and y)

        task_model_index = self.find_task_model(task_info['task_index'])
        self.sess.run(self.update[task_model_index], feed_dict={self.model_input: X, self.y_batch[task_model_index]: y, self.dropout_prob: 0.5})

    def _collect_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list