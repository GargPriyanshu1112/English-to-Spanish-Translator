
import tensorflow as tf


class Encoder(tf.keras.Model): 
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) 
        self.gru = tf.keras.layers.GRU(units=encoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    
    def call(self, x, initial_state):
        x = self.embedding(x)
        enc_hidden_states, enc_output_state = self.gru(x, initial_state=initial_state)
        return enc_hidden_states, enc_output_state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))



class BahdanauAttention(tf.keras.layers.Layer): 
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V  = tf.keras.layers.Dense(1)
    
    def call(self, enc_hidden_states,  dec_input_state):
        dec_input_state = tf.expand_dims(dec_input_state, axis=1)
        score = self.V(tf.nn.tanh(self.W1(dec_input_state) + self.W2(enc_hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights



class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = self.gru = tf.keras.layers.GRU(units=decoder_units,
                                                  return_sequences=True,
                                                  return_state=True,
                                                  recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")
        self.attention = BahdanauAttention(self.decoder_units)

    def call(self, x, enc_hidden_states, dec_input_state):
        context_vector, attention_weights = self.attention(enc_hidden_states, dec_input_state)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
