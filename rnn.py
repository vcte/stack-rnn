# Stack RNN and LSTM implementations, and recurrent wrapper class

import numpy as np
import tensorflow as tf

def matmul_broadcast(a, b):
    """Multiply all innermost 2x2 matrices of a and b.

       Args:
         a: (a_n, ..., a_2, a_1) tensor
         b: (b_n, ..., b_2, b_1) tensor, where a_1 = b_2

       Returns:
         (a_n, ..., a_3, b_n, ..., b_3, a_2, b_1) tensor
    """
    # get shape as list
    a_dims = [d if d != None else -1 for d in a.shape.as_list()]
    b_dims = [d if d != None else -1 for d in b.shape.as_list()]
    
    # reshape a to (a_n * ... * a_2, a_1)
    a_shape = np.array((-1, a_dims[-1]))
    a_ = tf.reshape(a, a_shape)

    # reshape b to (b_2, b_1 * b_n * ... * b_3)
    b_shape = np.array((b_dims[-2], -1))
    b_perm = [(i - 2 + len(b_dims)) % len(b_dims) for i in range(len(b_dims))]
    b_ = tf.reshape(tf.transpose(b, perm = b_perm), b_shape)

    # matrix product is (a_n * ... * a_2, b_1 * b_n * ... * b_3)
    a_b = tf.matmul(a_, b_)

    # refold to (a_n, ..., a_2, b_1, b_n, ..., b_3)
    a_b_shape = np.array(a_dims[: -1] + b_dims[-1 :] + b_dims[: -2])
    a_b = tf.reshape(a_b, a_b_shape)

    # permute to (a_n, ..., a_3, b_n, ..., b_3, a_2, b_1)
    a_b_perm = list(range(0, len(a.shape) - 2)) + \
               list(range(len(a.shape), len(a.shape) + len(b.shape) - 2)) + \
               list(range(len(a.shape) - 2, len(a.shape)))
    a_b = tf.transpose(a_b, perm = a_b_perm)
    
    return a_b

class StackRNNCell(tf.nn.rnn_cell.RNNCell):
    """StackRNN cell

    Implementation is based on: https://arxiv.org/abs/1503.01007
    """
    def __init__(self, num_units, no_op = False, n_stack = 1, k = 1,
                 stack_size = 200, mode = 1, 
                 activation = None, reuse = None, name = None):
        """Initialize the Stack RNN cell.

        Args:
          num_units: int, The number of hidden units in the cell.
          no_op: Bool, Whether to include no-op action. Default: False
          n_stack: int, number of stacks. Default 1
          k: int, number of items to read off the top of the stack. Default 1
          stack_size: int, number of elements in stack. Default: 200
          mode: int, switch between recurrence only through stacks (mode = 1)
                     and recurrence through hidden layer + stacks (mode = 2)
          activation: Activation function of inner states. Default: sigmoid
          reuse: (optional) Bool, whether to use variables in an existing scope. 
          name: String, the name of the layer.
        """
        super(StackRNNCell, self).__init__(_reuse = reuse, name = name)

        # Inputs must be 2-dimensional.
        self.input_spec = tf.contrib.keras.layers.InputSpec(ndim = 2)

        self._num_units = num_units
        self._no_op = no_op
        self._n_stack = n_stack
        self._k = k
        self._stack_size = stack_size
        self._mode = mode
        self._activation = activation or tf.sigmoid

    @property
    def state_size(self):
        return ([self._n_stack, self._stack_size + 1], [self._num_units])

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs_shape[1] to be known")
        input_depth = inputs_shape[1].value

        # hidden layer
        self._U = self.add_variable("U", shape = [input_depth, self._num_units])
        self._R = self.add_variable("R",
            shape = [self._num_units, self._num_units])
        self._P = self.add_variable("P",
            shape = [self._n_stack * self._k, self._num_units])
        self._bias_hidden = self.add_variable("bias_hidden",
            shape = [self._num_units],
            initializer = tf.zeros_initializer(dtype = self.dtype))

        # action layer
        n_actions = 3 if self._no_op else 2
        self._A = self.add_variable("A",
            shape = [self._n_stack, self._num_units, n_actions])
        self._bias_action = self.add_variable("bias_action",
            shape = [self._n_stack, n_actions],
            initializer = tf.zeros_initializer(dtype = self.dtype))

        # push to stack layer
        self._D = self.add_variable("D",
            shape = [self._n_stack, self._num_units, 1])
        self._bias_push = self.add_variable("bias_push",
            shape = [self._n_stack, 1],
            initializer = tf.zeros_initializer(dtype = self.dtype))

        self.built = True

    def call(self, inputs, state):
        """StackRNN cell.

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: Tuple of state tensors, first is tensor with shape
                 `[batch_size, n_stack, stack_size]`, second is tensor
                 with shape `[batch_size, state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state. 
        """
        PUSH, POP, NO_OP = 0, 1, 2
        s, h = state

        # compute new hidden state
        top_k = tf.reshape(s[:, :, 0 : self._k],
                           (-1, self._n_stack * self._k))
        h_in = [tf.matmul(inputs, self._U),
                tf.matmul(top_k, self._P)]
        if self._mode == 2:
            h_in.append(tf.matmul(h, self._R))
        new_h = self._activation(tf.add(tf.add_n(h_in), self._bias_hidden))

        # compute weights of each stack action
        swap_sb = lambda t: tf.transpose(t, perm = (1, 0, 2))
        a = tf.nn.softmax(tf.add(swap_sb(matmul_broadcast(new_h, self._A)),
                                 self._bias_action))

        # compute values to push on top of the stack
        d = self._activation(tf.add(swap_sb(matmul_broadcast(new_h, self._D)),
                                    self._bias_push))
        s_ = tf.concat((d, s), axis = 2)

        # weighted average of each stack operation
        ss = [tf.multiply(a[:, :, PUSH : PUSH + 1], s_[:, :, : -2]),
              tf.multiply(a[:, :, POP : POP + 1], s_[:, :, 2 :])]
        if self._no_op:
            ss.append(tf.multiply(a[:, :, NO_OP : NO_OP + 1], s_[:, :, 1 : -1]))

        # append sentinel value (-1) to bottom of stack
        new_s = tf.concat((tf.add_n(ss), s[:, :, -1 :]), axis = 2)

        return new_h, (new_s, new_h)

    def zero_state(self, batch_size, dtype):
        return (tf.zeros((batch_size, self._n_stack, self._stack_size + 1),
                         dtype = dtype) - 1, 
                tf.zeros((batch_size, self._num_units), dtype = dtype))

class StackLSTMCell(tf.nn.rnn_cell.RNNCell):
    """StackLSTM cell

    Implementation is based on: https://arxiv.org/abs/1503.01007
    """
    def __init__(self, num_units, no_op = False, n_stack = 1, k = 1,
                 stack_size = 200, 
                 activation = None, reuse = None, name = None):
        """Initialize the Stack LSTM cell.

        Args:
          num_units: int, The number of hidden units in the cell.
          no_op: Bool, Whether to include no-op action. Default: False
          n_stack: int, number of stacks. Default 1
          k: int, number of items to read off the top of the stack. Default 1
          stack_size: int, number of elements in stack. Default: 200
          activation: Activation function of inner states. Default: sigmoid
          reuse: (optional) Bool, whether to use variables in an existing scope. 
          name: String, the name of the layer.
        """
        super(StackLSTMCell, self).__init__(_reuse = reuse, name = name)

        # Inputs must be 2-dimensional.
        self.input_spec = tf.contrib.keras.layers.InputSpec(ndim = 2)

        self._num_units = num_units
        self._no_op = no_op
        self._n_stack = n_stack
        self._k = k
        self._stack_size = stack_size
        self._activation = activation or tf.sigmoid

    @property
    def state_size(self):
        return ([self._n_stack, self._stack_size + 1],
                [self._num_units], [self._num_units])

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs_shape[1] to be known")
        input_depth = inputs_shape[1].value
        init = tf.zeros_initializer(dtype = self.dtype)

        # lstm layer
        self._R = self.add_variable("R",
            shape = [input_depth + self._num_units, 4 * self._num_units])
        self._bias_lstm = self.add_variable("bias_lstm",
            shape = [4 * self._num_units],
            initializer = init)

        # hidden layer
        self._P = self.add_variable("P",
            shape = [self._n_stack * self._k, self._num_units])
        self._bias_hidden = self.add_variable("bias_hidden",
            shape = [self._num_units],
            initializer = init)

        # action layer
        n_actions = 3 if self._no_op else 2
        self._A = self.add_variable("A",
            shape = [self._n_stack, self._num_units, n_actions])
        self._bias_action = self.add_variable("bias_action",
            shape = [self._n_stack, n_actions],
            initializer = init)

        # push to stack layer
        self._D = self.add_variable("D",
            shape = [self._n_stack, self._num_units, 1])
        self._bias_push = self.add_variable("bias_push",
            shape = [self._n_stack, 1],
            initializer = init)

        self.built = True

    def call(self, inputs, state):
        """StackLSTM cell.

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: Tuple of state tensors, first is tensor with shape
                 `[batch_size, n_stack, stack_size]`, second and third are
                 tensors with shape `[batch_size, state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state. 
        """
        PUSH, POP, NO_OP = 0, 1, 2
        s, c, h = state

        # compute lstm state
        gate_in = tf.add(tf.matmul(tf.concat([inputs, h], axis = 1), self._R),
                         self._bias_lstm)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value = gate_in, num_or_size_splits = 4, axis = 1)

        forget_bias = tf.constant(1, dtype = f.dtype)
        new_c = tf.add(tf.multiply(c, tf.sigmoid(tf.add(f, forget_bias))),
                       tf.multiply(tf.sigmoid(i), self._activation(j)))
        lstm_h = tf.multiply(self._activation(new_c), tf.sigmoid(o))

        # compute new hidden state
        top_k = tf.reshape(s[:, :, 0 : self._k],
                           (-1, self._n_stack * self._k))
        new_h = self._activation(tf.add(tf.add(tf.matmul(top_k, self._P),
                                               lstm_h),
                                        self._bias_hidden))

        # compute weights of each stack action
        swap_sb = lambda t: tf.transpose(t, perm = (1, 0, 2))
        a = tf.nn.softmax(tf.add(swap_sb(matmul_broadcast(new_h, self._A)),
                                 self._bias_action))

        # compute values to push on top of the stack
        d = self._activation(tf.add(swap_sb(matmul_broadcast(new_h, self._D)),
                                    self._bias_push))
        s_ = tf.concat((d, s), axis = 2)

        # weighted average of each stack operation
        ss = [tf.multiply(a[:, :, PUSH : PUSH + 1], s_[:, :, : -2]),
              tf.multiply(a[:, :, POP : POP + 1], s_[:, :, 2 :])]
        if self._no_op:
            ss.append(tf.multiply(a[:, :, NO_OP : NO_OP + 1], s_[:, :, 1 : -1]))

        # append sentinel value (-1) to bottom of stack
        new_s = tf.concat((tf.add_n(ss), s[:, :, -1 :]), axis = 2)

        return new_h, (new_s, new_c, new_h)

    def zero_state(self, batch_size, dtype):
        return (tf.zeros((batch_size, self._n_stack, self._stack_size + 1),
                         dtype = dtype) - 1, 
                tf.zeros((batch_size, self._num_units), dtype = dtype),
                tf.zeros((batch_size, self._num_units), dtype = dtype))

class RecurrentWrapper:
    def __init__(self, cell, n_symbols = 2, sgd_lr = 0.01, hard_clip = 15.0):
        """n_symbols: number of output symbols
           sgd_lr: learning rate
           hard_clip: maximum absolute value of gradients
        """
        # Placeholder for the inputs in a given iteration (batch_size = 1)
        self.symbols = tf.placeholder(tf.float32, [None, None, n_symbols],
                                      name = "lstm_symbols")
        self.targets = tf.placeholder(tf.float32, [None, None, n_symbols],
                                      name = "lstm_targets")        

        # Initial state of the LSTM memory
        batch_size = tf.shape(self.symbols)[1]
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # Given inputs with shape (time, batch, input_size) outputs:
        #  - outputs: (time, batch, output_size)
        #  - states:  (time, batch, hidden_size)
        outputs, states = tf.nn.dynamic_rnn(cell, self.symbols,
                                            initial_state = self.initial_state,
                                            time_major = True)

        # add linear layer
        final_projection = lambda x: tf.contrib.layers.fully_connected(
            x, num_outputs = n_symbols, activation_fn = None)

        self.outputs = outputs = tf.map_fn(final_projection, outputs)

        # predicted symbol is symbol with max probability
        self.probs = tf.nn.softmax(self.outputs)
        self.preds = tf.argmax(self.outputs, axis = 2)

        # loss and optimization
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels = self.targets, logits = outputs, name = "cross_entropy")

        clip = lambda grad: tf.clip_by_value(grad, -hard_clip, +hard_clip) \
               if grad is not None else grad
        opt = tf.train.GradientDescentOptimizer(sgd_lr)
        gvs = opt.compute_gradients(self.loss)
        gvs_ = [(clip(grad), var) for grad, var in gvs]
        self.train_op = opt.apply_gradients(gvs_)
