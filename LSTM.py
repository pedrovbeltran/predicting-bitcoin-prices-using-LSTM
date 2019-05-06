import tensorflow as tf
import numpy as np

class LSTM_Net(object):
    def __init__(self, hidden_layer):
        self.hidden_layer = hidden_layer

    #function to compute the gate states
    def LSTM_cell(self, input, output, state):
            
        input_gate = tf.sigmoid(tf.matmul(input, self.weights_input_gate) + tf.matmul(output, self.weights_input_hidden) + self.bias_input)
            
        forget_gate = tf.sigmoid(tf.matmul(input, self.weights_forget_gate) + tf.matmul(output, self.weights_forget_hidden) + self.bias_forget)
            
        output_gate = tf.sigmoid(tf.matmul(input, self.weights_output_gate) + tf.matmul(output, self.weights_output_hidden) + self.bias_output)
            
        memory_cell = tf.tanh(tf.matmul(input, self.weights_memory_cell) + tf.matmul(output, self.weights_memory_cell_hidden) + self.bias_memory_cell)
            
        state = state * forget_gate + input_gate * memory_cell
            
        output = output_gate * tf.tanh(state)
        return state, output

    def run_net(self, batch_size, window_size, clip_margin, \
                learning_rate, epochs, step, scaled_data):
        
        #windowing the data with window_data function
        from organize_data import window_data
        X, y = window_data(scaled_data, window_size)

        #we now split the data into training and test set
        X_train  = np.array(X[:1018])
        y_train = np.array(y[:1018])

        X_test = np.array(X[1018:])
        y_test = np.array(y[1018:])
        
        hidden_layer = self.hidden_layer #How many units do we use in LSTM cell

        #we define the placeholders
        inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
        targets = tf.placeholder(tf.float32, [batch_size, 1])

        #weights and implementation of LSTM cell
        # LSTM weights

        #Weights for the input gate
        self.weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
        self.weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
        self.bias_input = tf.Variable(tf.zeros([hidden_layer]))

        #weights for the forgot gate
        self.weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
        self.weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
        self.bias_forget = tf.Variable(tf.zeros([hidden_layer]))

        #weights for the output gate
        self.weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
        self.weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
        self.bias_output = tf.Variable(tf.zeros([hidden_layer]))

        #weights for the memory cell
        self.weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev=0.05))
        self.weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
        self.bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

        #Output layer weigts
        weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev=0.05))
        bias_output_layer = tf.Variable(tf.zeros([1]))

        #we now define loop for the network
        outputs = []
        for i in range(batch_size): #Iterates through every window in the batch

            #for each batch I am creating batch_state as all zeros and output for that window which is all zeros at the beginning as well.
            batch_state = np.zeros([1, hidden_layer], dtype=np.float32) 
            batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
            
            #for each point in the window we are feeding that into LSTM to get next output
            for ii in range(window_size):
                batch_state, batch_output = self.LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
                
            #last output is conisdered and used to get a prediction
            outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)

        #we define the loss
        losses = []

        for i in range(len(outputs)):
            losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
            
        loss = tf.reduce_mean(losses)

        #we define optimizer with gradient clipping
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

        #we now train the network
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        for i in range(1, epochs+1):
            traind_scores = []
            ii = 0
            epoch_loss = []
            while(ii + batch_size) <= len(X_train):
                X_batch = X_train[ii:ii+batch_size]
                y_batch = y_train[ii:ii+batch_size]
                
                o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:X_batch, targets:y_batch})
                
                epoch_loss.append(c)
                traind_scores.append(o)
                ii += batch_size
            if (i % step) == 0 or i == 1:
                print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))