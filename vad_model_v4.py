from __future__ import unicode_literals

import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv
import time
import json
import yaml
import logging
import seaborn as sns

from datetime import datetime
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell, GRUCell

plt.style.use('ggplot')

class VADModel(object):
  """ Building the Recurrent Neural Network model for Voice Activity Detection
  """

  @classmethod
  def build(cls, 
    param_dir):
    """
    Restore a previously trained model
    """
    with open(cls._parameters_file(param_dir)) as f:
      parameters = json.load(f)

      # Encapsulate training parameters
      training_parameters = TrainingParameters(parameters["training_epochs"])

      # Encapsulate model hyperparameters
      model_parameters = ModelParameters(
        parameters["learning_rate"],
        parameters["momentum"],
        parameters["model"],
        parameters["input_keep_probability"],
        parameters["output_keep_probability"],
        parameters["sequence_length"],
        parameters["input_dimension"],
        parameters["batch_size"], 
        parameters["state_size"], 
        parameters["n_layers"],
        parameters["n_classes"],
        parameters["threshold"],
        parameters["pk_step"],
        parameters["ma_step"])

      # Encapsulate directories name
      directories = Directories(parameters["log_dir"],
        parameters["checkpoint_linear_dir"])

      model = cls(
        model_parameters,
        training_parameters,
        directories)

    return model

  @classmethod
  def restore(cls, 
    session, 
    param_dir):
    """
    Restore a previously trained model and its session
    """
    with open(cls._parameters_file(param_dir)) as f:
      parameters = json.load(f)

      # Encapsulate training parameters
      training_parameters = TrainingParameters(parameters["training_epochs"])

      # Encapsulate model hyperparameters
      model_parameters = ModelParameters(
        parameters["learning_rate"],
        parameters["momentum"],
        parameters["model"],
        parameters["input_keep_probability"],
        parameters["output_keep_probability"],
        parameters["sequence_length"],
        parameters["input_dimension"],
        parameters["batch_size"], 
        parameters["state_size"], 
        parameters["n_layers"],
        parameters["n_classes"],
        parameters["threshold"],
        parameters["pk_step"],
        parameters["ma_step"])

      # Encapsulate directories name
      directories = Directories(parameters["log_dir"],
        parameters["checkpoint_linear_dir"])

      model = cls(
        model_parameters,
        training_parameters,
        directories)

      # Load the saved meta graph and restore variables
      checkpoint_file = tf.train.latest_checkpoint(directories.checkpoint_linear_dir)
      print("restoring graph from {} ...".format(checkpoint_file))
      # Restore an empty computational graph
      #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
      
      # Restore an already existing graph
      saver = tf.train.Saver()
      saver.restore(session, checkpoint_file)

    return model

  @staticmethod
  def _parameters_file(param_dir):
    return os.path.join(param_dir, "parameters.json")

  @staticmethod
  def _model_file(model_dir):
    return os.path.join(model_directory, "model")

  # this is a simpler version of Tensorflow's 'official' version. See:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
  @staticmethod
  def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon=1e-3):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    def fn1():
      batch_mean, batch_var = tf.nn.moments(inputs,[0])
      train_mean = tf.assign(pop_mean,
        pop_mean * decay + batch_mean * (1 - decay))
        
      train_var = tf.assign(pop_var,
        pop_var * decay + batch_var * (1 - decay))
        
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs,
          batch_mean, batch_var, beta, scale, epsilon)

    def fn2():
      return tf.nn.batch_normalization(inputs,
        pop_mean, pop_var, beta, scale, epsilon)

    result = tf.cond(is_training,lambda: fn1(), lambda: fn2())
    return result


  def while_condition(self, loop_idx, *_):
    print(tf.shape(self.h)[0])
    return tf.less(loop_idx, tf.shape(self.h)[0]-self.model_parameters.pk_step)
  
     
  def pk_indicator(self, i, j, segmenter ):
    def same():
      return tf.constant(1, dtype=tf.int32, shape=[])

    def different():
      return tf.constant(0, dtype=tf.int32, shape=[])

    is_equal = tf.equal(segmenter[i], segmenter[j])
    result = tf.cond(is_equal,lambda: same(), lambda: different())
    return result
    

  def while_body(self, loop_idx, temp_pk_miss, temp_pk_falsealarm):

    pk_miss = tf.multiply(self.pk_indicator(loop_idx, loop_idx+self.model_parameters.pk_step, self.h),
      tf.subtract(self.dump_one, self.pk_indicator(loop_idx, loop_idx+self.model_parameters.pk_step, self.r)),
      name='pk_miss')

    temp_pk_miss = tf.concat(
      values=[
      temp_pk_miss,
      [pk_miss]],
      axis=0,
      name='temp_pk_miss')

    pk_falsealarm = tf.multiply(self.pk_indicator(loop_idx, loop_idx+self.model_parameters.pk_step, self.r),
      tf.subtract(self.dump_one, self.pk_indicator(loop_idx, loop_idx+self.model_parameters.pk_step, self.h)),
      name='pk_falsealarm')

    temp_pk_falsealarm = tf.concat(
      values=[
      temp_pk_falsealarm,
      [pk_falsealarm]],
      axis=0,
      name='temp_pk_falsealarm')
        
        
    j = tf.add(loop_idx, 1, name="loop_idx_increment")
    return j, temp_pk_miss, temp_pk_falsealarm



  def ma_while_body(self, loop_idx, smoothed_output):
    return tf.add(loop_idx, 1), tf.concat([smoothed_output,
      tf.expand_dims(tf.reduce_mean(self.tmp_smoothed_predictions[loop_idx:loop_idx+self.ma_step]), 0)],
      axis=0)

  @staticmethod
  def variable_summaries(var, scope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram(scope, var)

  def __init__(self,
    model_parameters,
    training_parameters,
    directories, 
    **kwargs):

    """ Initialization of the RNN Model as TensorFlow computational graph
    """

    self.model_parameters = model_parameters
    self.training_parameters = training_parameters
    self.directories = directories

    # Define model hyperparameters Tensors
    with tf.name_scope("Parameters"):
      self.learning_rate = tf.placeholder(tf.float32, 
        name="learning_rate")
      self.momentum = tf.placeholder(tf.float32, 
        name="momentum")
      self.input_keep_probability = tf.placeholder(tf.float32, 
        name="input_keep_probability")
      self.output_keep_probability = tf.placeholder(tf.float32, 
        name="output_keep_probability")


      self.is_training = tf.placeholder(tf.bool)

    # Define input, output and initialization Tensors
    with tf.name_scope("Input"):
      self.inputs = tf.placeholder("float", [None, 
        self.model_parameters.sequence_length, 
        self.model_parameters.input_dimension], 
        name='input_placeholder')

      self.targets = tf.placeholder("float", [None, 
        self.model_parameters.sequence_length, 
        1], 
        name='labels_placeholder')

      self.init = tf.placeholder(tf.float32, shape=[None, 
        self.model_parameters.state_size], 
        name="init")

    # Define the TensorFlow RNN computational graph
    with tf.name_scope("LSTMRNN_RNN"):
      cells = []

      # Define the layers
      for _ in range(self.model_parameters.n_layers):
        if self.model_parameters.model == 'rnn':
          cell = BasicRNNCell(self.model_parameters.state_size)
        elif self.model_parameters.model == 'gru':
          cell = GRUCell(self.model_parameters.state_size)
        elif self.model_parameters.model == 'lstm':
          cell = BasicLSTMCell(self.model_parameters.state_size, state_is_tuple=True)
        elif self.model_parameters.model == 'nas':
          cell = NASCell(self.model_parameters.state_size)
        else:
          raise Exception("model type not supported: {}".format(self.model_parameters.model))

        if (self.model_parameters.output_keep_probability < 1.0 
          or self.model_parameters.input_keep_probability < 1.0):

          if self.model_parameters.output_keep_probability < 1.0 :
            cell = DropoutWrapper(cell,
              output_keep_prob=self.output_keep_probability)

          if self.model_parameters.input_keep_probability < 1.0 :
            cell = DropoutWrapper(cell,
              input_keep_prob=self.input_keep_probability)

        cells.append(cell)
      cell = MultiRNNCell(cells)

      # Simulate time steps and get RNN cell output
      self.outputs, self.next_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype = tf.float32)


    # Define cost Tensors
    with tf.name_scope("LSTMRNN_Cost"):

      # Flatten to apply same weights to all time steps
      self.flattened_outputs = tf.reshape(self.outputs, [-1, 
        self.model_parameters.state_size], 
        name="flattened_outputs")

      self.output_w = tf.Variable(tf.truncated_normal([
        self.model_parameters.state_size, 
        1], stddev=0.01), 
        name="output_weights")

      self.variable_summaries(self.output_w, 'output_weights')

      self.output_b = tf.Variable(tf.constant(0.1), 
        name="output_biases")

      self.variable_summaries(self.output_w, 'output_biases')

      # Define decision threshold Tensor
      self.decision_threshold = tf.Variable(self.model_parameters.threshold, 
        name="decision_threshold")

      # Define moving average step Tensor
      self.ma_step = tf.Variable(self.model_parameters.ma_step, 
        name="ma_step")

      # Softmax activation layer, using RNN inner loop last output
      # logits and labels must have the same shape [batch_size, num_classes]

      self.logits = tf.add(tf.matmul(self.flattened_outputs, self.output_w),
        self.output_b, 
        name="logits")

      self.logits_bn = self.batch_norm_wrapper(inputs=self.logits, 
        is_training=self.is_training)

      tf.summary.histogram('logits', self.logits)
      tf.summary.histogram('logits_bn', self.logits_bn)

      self.predictions = tf.reshape(self.logits, 
        [-1, self.model_parameters.sequence_length, 1], 
        name="predictions")

      self.shaped_predictions = tf.reshape(self.predictions, 
        [-1], 
        name="shaped_predictions")

      self.tmp_smoothed_predictions = tf.concat([self.shaped_predictions,
        tf.fill(tf.expand_dims(self.ma_step-1, 0), self.shaped_predictions[tf.shape(self.shaped_predictions)[0]-1])],
        axis=0,
        name="tmp_smoothed_predictions")

      self.ma_loop_idx = tf.constant(0, dtype='int32')
      self.shaped_smoothed_predictions = tf.zeros([0], dtype='float32')

      _, self.shaped_smoothed_predictions = tf.while_loop(lambda i, _: i < tf.shape(self.shaped_predictions)[0],
        self.ma_while_body,
        [self.ma_loop_idx, self.shaped_smoothed_predictions],
        shape_invariants=[tf.TensorShape([]),
        tf.TensorShape([None])])

      self.smoothed_predictions = tf.reshape(self.shaped_smoothed_predictions, 
        [-1, self.model_parameters.sequence_length, 1], 
        name="smoothed_predictions")


      self.soft_predictions_summary = tf.summary.tensor_summary("soft_predictions", self.smoothed_predictions)
      # self.soft_predictions_summary = tf.summary.tensor_summary("soft_predictions", self.predictions)

      # self.shaped_logits = tf.reshape(self.logits, 
      #   [-1, self.model_parameters.sequence_length, 1], 
      #   name="shaped_logits")

      # Cross-Entropy
      # self.cost = tf.reduce_mean(-tf.reduce_sum(
      #   self.targets * tf.log(self.predictions), 
      #   reduction_indices=[2]), name="cross_entropy")


      # self.cross_entropy = tf.reduce_mean(
      #   tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None,
      #     labels=self.targets,
      #     logits=self.predictions),
      #   name="cross_entropy")

      # self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      #   _sentinel=None,
      #   labels=self.targets,
      #   logits=self.shaped_logits,
      #   name="cross_entropy")

      # Root Mean Squared Error
      # self.mean_squared_error = tf.losses.mean_squared_error(
      #   labels=self.targets,
      #   predictions=self.predictions)

      self.cost = tf.sqrt(tf.reduce_mean(
        tf.squared_difference(
          self.smoothed_predictions, self.targets)))

      # self.cost = tf.sqrt(tf.reduce_mean(
      #   tf.squared_difference(
      #     self.predictions, self.targets)))

      tf.summary.scalar('training_cost', self.cost)

      # self.cost = tf.reduce_mean(
      #   self.cross_entropy,
      #   name="cost")

      voicing_condition = tf.greater(self.smoothed_predictions, 
        tf.fill(tf.shape(self.smoothed_predictions), self.decision_threshold),
        name="thresholding")

      # voicing_condition = tf.greater(self.predictions, 
      #   tf.fill(tf.shape(self.predictions), self.decision_threshold),
      #   name="thresholding")


      self.label_predictions = tf.where(voicing_condition, 
        tf.ones_like(self.smoothed_predictions) , 
        tf.zeros_like(self.smoothed_predictions),
        name="label_predictions")

      # self.label_predictions = tf.where(voicing_condition, 
      #   tf.ones_like(self.predictions) , 
      #   tf.zeros_like(self.predictions),
      #   name="label_predictions")

      self.hard_predictions_summary = tf.summary.tensor_summary("hard_predictions", self.label_predictions)


      self.correct_prediction = tf.equal(self.label_predictions, 
        self.targets, 
        name="correct_predictions")

      self.r = tf.reshape(self.targets, [-1])
      self.h = tf.reshape(self.label_predictions, [-1])

      # Defined outside the while loop to avoid problems
      self.dump_one = tf.constant(1, dtype=tf.int32, shape=[])

      self.temp_pk_miss = tf.Variable([0], tf.int32, name='temp_pk_miss')
      self.temp_pk_falsealarm = tf.Variable([0], tf.int32, name='temp_pk_falsealarm')
      self.loop_idx = tf.constant(0, dtype=tf.int32, name='loop_idx')
        
      self.loop_vars = self.loop_idx, self.temp_pk_miss, self.temp_pk_falsealarm
        
      _, self.all_temp_pk_miss,  self.all_temp_pk_falsealarm = tf.while_loop(
        self.while_condition,
        self.while_body,
        self.loop_vars,
        shape_invariants=(self.loop_idx.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])))
        
      self.pk_miss = tf.reduce_mean(
        tf.cast(self.all_temp_pk_miss, tf.float32))
      tf.summary.scalar('p_miss', self.pk_miss)
      
      self.pk_falsealarm = tf.reduce_mean(
        tf.cast(self.all_temp_pk_falsealarm, tf.float32))
      tf.summary.scalar('p_falsealarm', self.pk_falsealarm)

      self.pk = tf.reduce_mean(
        tf.cast(
          tf.add(self.all_temp_pk_miss, self.all_temp_pk_falsealarm), 
          tf.float32),
        name='pk')

      tf.summary.scalar('pk', self.pk)
      
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), 
        name="accuracy")

      tf.summary.scalar('accuracy', self.accuracy)

      self.recall, self.update_op_recall = tf.metrics.recall(
        labels=self.targets,
        predictions=self.label_predictions,
        name="recall")

      tf.summary.scalar('recall', self.recall)

      self.precision, self.update_op_precision = tf.metrics.precision(
        labels=self.targets,
        predictions=self.label_predictions,
        name="precision")

      tf.summary.scalar('precision', self.precision)



    # Define Training Tensors
    with tf.name_scope("LSTMRNN_Train"):

      # Momentum optimisation
      self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
        momentum=self.momentum, 
        name="optimizer")


      self.train_step = self.optimizer.minimize(self.cost, 
        name="train_step")

      # Initializing the variables
      self.initializer = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())
      # self.local_initializer = tf.local_variables_initializer()
      # self.global_initializer = tf.global_variables_initializer()
      #self.summary = tf.merge_all_summaries()

  # def weight_variable(shape):
  #   initial = tf.truncated_normal(shape, stddev=0.1)
  #   return tf.Variable(initial)

  # def bias_variable(shape):
  #   initial = tf.constant(0.1, shape=shape)
  #   return tf.Variable(initial)

  # def conv2d(x, W):
  #   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # def max_pool_2x2(x):
  #   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
  #     strides=[1, 2, 2, 1], padding='SAME')

  @property
  def batch_size(self):
    return self.inputs.get_shape()[0].value

  @property
  def sequence_length(self):
    return self.inputs.get_shape()[1].value

  @property
  def input_dimension(self):
    return self.inputs.get_shape()[2].value

  @property
  def n_classes(self):
    return self.targets.get_shape()[2].value

  @property
  def state_size(self):
    return self.init.get_shape()[1].value

  @staticmethod
  def perplexity(cost, iterations):
    return np.exp(cost / iterations)

  @staticmethod
  def plot_training_losses(cost_history, training_epochs):
    fig = plt.figure(figsize=(15,10))
    plt.plot(cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plot/losses_linear_'+timestr+'.png', bbox_inches='tight')
    # plt.show()

  @staticmethod
  def plot_metrics(metric, training_epochs):
    fig = plt.figure(figsize=(15,10))
    plt.plot(metric, label='Training accuracies')
    plt.title('Accuracy evaluation summary ')
    plt.axis([0,training_epochs,0,np.max(metric)+0.1])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plot/accuracies_linear_'+timestr+'.png')
    # plt.show()

  @staticmethod
  def plot_predictions(predictions):
    fig = plt.figure(figsize=(15,10))
    plt.plot(predictions, color="blue")
    plt.title('Output predictions')
    plt.axis([0,len(predictions),0,np.max(predictions)+0.2])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plot/prediction_linear_'+timestr+'.png')
    # plt.show()

  @staticmethod
  def plot_targets(targets):
    fig = plt.figure(figsize=(15,10))
    plt.plot(targets, color="blue")
    plt.title('Ground Truth')
    plt.axis([0,len(targets),0,np.max(targets)+0.2])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plot/target_linear_'+timestr+'.png')
    # plt.show()

  @staticmethod
  def plot_prediction_summary(predictions, ground_truth):
    plt.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
    fig, ax = plt.subplots()

    # ax.plot(range(0,len(predictions)*500,500),predictions, label='Predictions')
    # ax.plot(range(0,len(ground_truth)*500,500),ground_truth, label='Ground Truth')

    ax.plot(predictions, label='Predictions')
    ax.plot(ground_truth, label='Ground Truth')

    ax.set_xlabel('Test data')
    ax.set_ylabel('Voicing score')
    # ax.set_ylim([0.8,1]) # Cut the y axes and only show the ones 0.8 to 1
    ax.set_title('Predictions evaluation summary ')
    ax.legend(loc=4)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plot/summary_linear_'+timestr+'.png')
    # plt.show()


  def _get_batch(self,
    X_train, 
    Y_train):
    """
    Formatting our raw data s.t. [batch_size, sequence_length, input_dimension]
    :param X_train: dataset features matrix
    :type 2-D Numpy array
    :param Y_train: dataset one-hot encoded labels matrix
    :type 2-D Numpy array
    :return: Iteratot over training batches
    :rtype: Iterator
    """

    raw_data_length = len(X_train)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = raw_data_length // self.model_parameters.batch_size
    data_x = np.zeros([self.model_parameters.batch_size, 
      batch_partition_length, 
      self.model_parameters.input_dimension], 
      dtype=np.float32)

    data_y = np.zeros([self.model_parameters.batch_size, 
      batch_partition_length, 
      1], 
      dtype=np.float32)
    #data_y = np.zeros([batch_size, n_classes], dtype=np.int32)
    
    for i in range(self.model_parameters.batch_size):
        data_x[i] = X_train[batch_partition_length * i:batch_partition_length * (i + 1), :]
        data_y[i] = Y_train[batch_partition_length * i:batch_partition_length * (i + 1),:]
    
    # further divide batch partitions into sequence_length for truncated backprop
    epoch_size = batch_partition_length // self.model_parameters.sequence_length

    for i in range(epoch_size):
        x = data_x[:, i * self.model_parameters.sequence_length:(i + 1) * self.model_parameters.sequence_length,:]
        y = data_y[:, i * self.model_parameters.sequence_length:(i + 1) * self.model_parameters.sequence_length,:]
        yield (x, y)



  def _get_epochs(self, 
    n, 
    X_train, 
    Y_train):
    """
    Generate iterator over training epochs
    :param n: max number of training epochs
    :type int
    :param X_train: dataset features matrix
    :type 2-D Numpy array
    :param Y_train: dataset one-hot encoded labels matrix
    :type 2-D Numpy array
    :return: Iteratot over training epochs
    :rtype: Iterator
    """
    for i in range(n):
        yield self._get_batch(X_train, Y_train)


  def train(self,
    session,
    X_train,
    Y_train,
    checkpoint_every=1000,
    log_dir = 'log',
    display_step=5,
    verbose=True):

    """ Training the network
    :param X_train: features matrix
    :type 2-D Numpy array of float values
    :param Y_train: one-hot encoded labels matrix
    :type 2-D Numpy array of int values
    :param checkpoint_every: RNN model checkpoint frequency 
    :type int 
    :param log_dir: TensorBoard log directory
    :type string
    :param display_step: number of traing epochs executed before logging messages
    :type int
    :param verbose: display log mesages on screen at each training epoch
    :type boolean
    :returns: Cost history of each training epoch
              and the training Perplexity
    :rtype float, float
    :raises: -
    """

    print("\nTraining the network...\n")

    epoch_cost=0
    epoch_accuracy=0
    epoch_recall=0
    epoch_precison=0
    epoch_iteration=0

    winner_accuracy=0
    winner_recall=0
    winner_since=0

    current_epoch=0
    current_iteration=0
    done = False

    cost_history = np.empty(shape=[1], dtype=float)
    perplexity_history = np.empty(shape=[1], dtype=float)
    accuracy_history = np.empty(shape=[1], dtype=float)
    recall_history = np.empty(shape=[1], dtype=float)
    precision_history = np.empty(shape=[1], dtype=float)

    try:
    #with tf.Session() as session:

      # Merge all the summaries and write them out
      self.summary = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      test_writer = tf.summary.FileWriter(os.path.join(log_dir,'test', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

      writer = tf.summary.FileWriter(os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      writer.add_graph(session.graph)

      session.run(self.initializer)

      saver = tf.train.Saver(tf.global_variables())

      for epoch_idx, epoch in enumerate(
        self._get_epochs(
          self.training_parameters.training_epochs, 
          X_train, 
          Y_train)):

        current_epoch = epoch_idx
        avg_cost = 0.

        #training_state = np.zeros((batch_size, state_size))

        current_iteration = 0
        done = False

        for batch_step, (batch_x, batch_y) in enumerate(epoch):
          
          current_iteration = batch_step

          if epoch_idx % 100 == 99 and batch_step == 0: # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Run optimization op (backprop) and cost op (to get loss value)
            _summary, _train_step, _cost, _prediction_series, _label_predictions, _unshaped_predictions = session.run(
              [self.summary, self.train_step, self.cost, self.smoothed_predictions, self.label_predictions, self.logits], 
              feed_dict={ 
              self.inputs:batch_x, 
              self.targets:batch_y,
              self.learning_rate : self.model_parameters.learning_rate,
              self.momentum : self.model_parameters.momentum,
              self.input_keep_probability : self.model_parameters.input_keep_probability,
              self.output_keep_probability : self.model_parameters.output_keep_probability,
              self.is_training : True},
              options=run_options,
              run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch_idx)
            train_writer.add_summary(_summary, batch_step)

          else:

            # Run optimization op (backprop) and cost op (to get loss value)
            _summary, _train_step, _cost, _prediction_series, _label_predictions, _unshaped_predictions = session.run(
              [self.summary, self.train_step, self.cost, self.smoothed_predictions, self.label_predictions, self.logits], 
              feed_dict={ 
              self.inputs:batch_x, 
              self.targets:batch_y,
              self.learning_rate : self.model_parameters.learning_rate,
              self.momentum : self.model_parameters.momentum,
              self.input_keep_probability : self.model_parameters.input_keep_probability,
              self.output_keep_probability : self.model_parameters.output_keep_probability,
              self.is_training : True})

            train_writer.add_summary(_summary, batch_step)
          
          # Compute average loss 
          avg_cost += _cost / self.model_parameters.batch_size
          tf.summary.scalar('train_loss', avg_cost)

          if (epoch_idx * self.model_parameters.batch_size + batch_step) % checkpoint_every == 0 or (
            epoch_idx == self.training_parameters.training_epochs-1 and 
            batch_step == self.model_parameters.batch_size-1):

            # Save for the last result
            checkpoint_path = os.path.join(self.directories.checkpoint_linear_dir, 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=epoch_idx * self.model_parameters.batch_size + batch_step)
            print("model saved to {}".format(checkpoint_path))


          epoch_cost += _cost
          epoch_iteration += self.model_parameters.batch_size

          # Display logs per epoch step
          if epoch_idx % display_step == 0:
            if verbose and not done:
              # Calculate batch accuracy
              epoch_accuracy, epoch_recall, epoch_update_op_recall, epoch_precision, epoch_update_op_precision, _threshold = session.run(
                [self.accuracy, self.recall, self.update_op_recall, self.precision, self.update_op_precision, self.decision_threshold], 
                feed_dict= {
                self.inputs: batch_x, 
                self.targets: batch_y,
                self.is_training : True})
              time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
              done = True

              test_writer.add_summary(_summary, epoch_idx)

              print(str(time),
                ": Epoch:", '%04d' % (epoch_idx),
                "cost=", "{:.9f}".format(avg_cost),
                ", Accuracy= ", "{:.5f}".format(epoch_accuracy),
                ", Recall= ", "{:.5f}".format(epoch_recall),
                ", Precision= ", "{:.5f}".format(epoch_precision),
                ", Threshold= ", _threshold)

        cost_history = np.append(cost_history,avg_cost) # Epoch cost
        accuracy_history = np.append(accuracy_history,epoch_accuracy) # Epoch accuracy
        recall_history = np.append(recall_history,epoch_recall)
        precision_history = np.append(precision_history,epoch_precision)

        if (winner_recall<epoch_recall):
          winner_recall=epoch_recall
          winner_since=0
        else:
          winner_since=winner_since+1

        if (winner_since>=20):
          raise Exception('No Recall improvements since 20 epochs ... Force stopping!') 

    except KeyboardInterrupt:
      pass

    except Exception as error:
      train_writer.close()
      print('Early stopping mechanism enabled ...')
      print(error)
      pass

    print("Stop training at epoch %d, iteration %d" % (current_epoch, current_iteration),
                ", Accuracy= ", "{:.5f}".format(epoch_accuracy),
                ", Recall= ", "{:.5f}".format(epoch_recall),
                ", Precision= ", "{:.5f}".format(epoch_precision))
                # "Perplexity=", "{:.4f}".format(self.perplexity(epoch_cost, epoch_iteration)),
                

    #logger.info("Stop training at epoch %d, iteration %d" % (current_epoch, current_iteration))
    #summary.close()
    #logger.info("Saved model in %s " % self.directories.checkpoint_linear_dir)

    print("Optimization Finished!")
    # self.plot_training_losses(cost_history)

    # try:

    #   self.plot_training_losses(cost_history, current_epoch)
    #   print("Training losses plotted in plot folder")

    #   self.plot_metrics(accuracy_history, current_epoch)
    #   print("Training metrics plotted in plot folder")

    # except Exception as e:
    #   print("ERROR Exception while plotting !")
    #   print(e)
    #   pass

    return cost_history, epoch_accuracy, epoch_update_op_recall, epoch_update_op_precision


  def _write_model_parameters(self, param_dir):
    """ Store parameters in a JSON file
    :param param_dir: parameter save directory
    :type string
    """
    parameters = {
    "training_epochs" : self.training_parameters.training_epochs,
    "learning_rate" : self.model_parameters.learning_rate,
    "momentum" : self.model_parameters.momentum,
    "model" : self.model_parameters.model,
    "input_keep_probability" : self.model_parameters.input_keep_probability,
    "output_keep_probability" : self.model_parameters.output_keep_probability,
    "sequence_length" : self.model_parameters.sequence_length,
    "input_dimension" : self.model_parameters.input_dimension,
    "batch_size" : self.model_parameters.batch_size,
    "state_size" : self.model_parameters.state_size,
    "n_layers" : self.model_parameters.n_layers,
    "n_classes" : self.model_parameters.n_classes,
    "threshold" : self.model_parameters.threshold,
    "log_dir" : self.directories.log_dir,
    "checkpoint_linear_dir" : self.directories.checkpoint_linear_dir,
    "pk_step" : self.model_parameters.pk_step,
    "ma_step" : self.model_parameters.ma_step,
    }

    with open(self._parameters_file(param_dir), "w") as f:
      json.dump(parameters, f, indent=4)


  def evaluate(self,
    session,
    X_test, 
    Y_test,
    allow_soft_placement=True,
    log_device_placement=False):
    """ Evaluating the network
    :param X_test: features matrix
    :type 2-D Numpy array of float values
    :param Y_test: one-hot encoded labels matrix
    :type 2-D Numpy array of int values
    :returns: -
    :raises: -
    """

    print("\nEvaluating the network...\n")


    
    self.evaluate_summary = tf.summary.merge([self.hard_predictions_summary, self.soft_predictions_summary])
    evaluate_writer = tf.summary.FileWriter(os.path.join(self.directories.log_dir,'evaluate', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    session.run(tf.local_variables_initializer())
    
    raw_data_length = len(X_test)
    sequence_size = raw_data_length // self.model_parameters.sequence_length
    epoch_cost = epoch_iteration = 0
    
    data_x_test = np.zeros([
      sequence_size,
      self.model_parameters.sequence_length,
      self.model_parameters.input_dimension],
      dtype=np.float32)

    data_y_test = np.zeros([
      sequence_size,
      self.model_parameters.sequence_length,
      1],
      dtype=np.float32)
        
    for i in range(sequence_size):
      data_x_test[i] = X_test[self.model_parameters.sequence_length * i:self.model_parameters.sequence_length * (i + 1), :]
      data_y_test[i] = Y_test[self.model_parameters.sequence_length * i:self.model_parameters.sequence_length * (i + 1), :]
                
    _summary, _cost, _all_predictions, _label_predictions = session.run(
      [self.evaluate_summary, self.cost, 
      self.smoothed_predictions, self.label_predictions],
      feed_dict={
      self.inputs: data_x_test,
      self.targets: data_y_test,
      self.input_keep_probability : self.model_parameters.input_keep_probability,
      self.output_keep_probability : self.model_parameters.output_keep_probability,
      self.is_training : False})

    epoch_cost += _cost
    epoch_iteration += sequence_size

    evaluate_writer.add_summary(_summary)

    # self.plot_predictions(label_predictions)

    # Print accuracy if test label set is provided
    if Y_test is not None:
        evaluation_accuracy, evaluation_recall, evaluation_update_op__recall, evaluation_precision, evaluation_update_op__precision = session.run(
            [self.accuracy, self.recall, self.update_op_recall, self.precision, self.update_op_precision],  
            feed_dict={
            self.inputs: data_x_test, 
            self.targets: data_y_test,
            self.input_keep_probability : self.model_parameters.input_keep_probability,
            self.output_keep_probability : self.model_parameters.output_keep_probability,
            self.is_training : False})
    
    # evaluation_perplexity=self.perplexity(epoch_cost, epoch_iteration)

    print("Total number of test examples: {}".format(len(Y_test)))
    print("Accuracy: ",evaluation_accuracy)
    print("Recall: ",evaluation_update_op__recall)
    print("Precision: ",evaluation_update_op__precision)
    
    # Save the results in a CSV output file
    out_path = "prediction_linear_soft.csv"
    out_path2 = "prediction_linear_hard.csv"

    print("Saving evaluation to {0}".format(out_path))

    _hard_predictions = np.reshape(_label_predictions, [-1,1])
    _soft_predictions = np.reshape(_all_predictions, [-1,1])

    print(_hard_predictions.shape)
    print(_soft_predictions.shape)

    try:
      self.plot_predictions(_hard_predictions[390000:398000,:])
      print("Predictions plotted in plot folder")

      self.plot_targets(Y_test[390000:398000,:])
      print("Predictions plotted in plot folder")

      self.plot_prediction_summary(predictions=_soft_predictions[390000:398000,:],
        ground_truth=Y_test[390000:398000,:])
      print("Evaluation summary plotted in plot folder ... I SEE WHAT YOU DID THERE!")

    except Exception as e:
      print("ERROR Exception while plotting !")
      print(e)
      pass

    with open(out_path, 'w') as f:
      csv.writer(f).writerows(np.reshape(_soft_predictions, [-1]))

    with open(out_path2, 'w') as f:
      csv.writer(f).writerows(np.reshape(_hard_predictions, [-1]))

    return evaluation_accuracy, evaluation_update_op__recall, evaluation_update_op__precision




  @staticmethod
  def summary_writer(summary_directory, session):
    class NullSummaryWriter(object):
      def add_summary(self, *args, **kwargs):
        pass

      def flush(self):
        pass

      def close(self):
        pass

    if summary_directory is not None:
      return tf.train.SummaryWriter(summary_directory, session.graph)
    else:
      return NullSummaryWriter()



# Objects used to store parameters

class TrainingParameters(object):
    def __init__(self, 
      training_epochs):
      """ Encapsulation of RNN training parameters
      :param training_epochs: number of the training epochs
      :type int
      """
      self.training_epochs = training_epochs


class ModelParameters(object):
    def __init__(self,
      learning_rate,
      momentum=None,
      model='lstm',
      input_keep_probability=1.0,
      output_keep_probability=1.0,
      sequence_length=None,
      input_dimension=None,
      batch_size=None, 
      state_size=None, 
      n_layers=None,
      n_classes=None,
      threshold=0.5,
      pk_step=50,
      ma_step=10):

      """ Encapsulation of RNN model hyperparameters
      :param learning_rate: gradient descent optimization learning rate
      :type float between 0..1
      :param momentum: momentum optimization parameter
      :type float between 0..1
      :param input_keep_probability:
      :type float between 0..1
      :param output_keep_probability: 
      :type float between 0..1
      :param sequence_length: number of truncated backpropagation steps
      :type int
      :param input_dimension: number of features
      :type int
      :param batch_size: size of the training batch
      :type int
      :param state_size: number of LSTM memory cells inside each memory block
      :type int
      :param n_layers: number of layers in the RNN
      ;type int
      :param n_classes: number of class labels
      :type int
      """

      self.learning_rate = learning_rate
      self.momentum = momentum
      self.model=model
      self.input_keep_probability = input_keep_probability
      self.output_keep_probability = output_keep_probability
      self.sequence_length=sequence_length
      self.input_dimension=input_dimension
      self.batch_size=batch_size
      self.state_size=state_size
      self.n_layers=n_layers
      self.n_classes=n_classes
      self.threshold=threshold
      self.pk_step=pk_step
      self.ma_step=ma_step


class Directories(object):
    def __init__(self, 
      log_dir, 
      checkpoint_linear_dir):
      """ Encaplsulation of the directories names
      :param log_dir: TensorBoard log directory
      :type string
      :param checkpoint_linear_dir: TensorFlow checkpoint directory
      :type string
      """
      self.log_dir = log_dir
      self.checkpoint_linear_dir = checkpoint_linear_dir



