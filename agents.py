import tensorflow as tf
import tensorflow_probability as tfp

class Target_Distractor_Module(tf.keras.layers.layer)
    """
    Module that creates an informed sender (See Lazaridou 2016) takes in a target and distractors and returns a single embedding (contrary to the token directly)
    """
    def __init__(self, reference_object_size, num_distractors, hidden_size, num_conv_layers, num_conv_filters, is_residual=True, embedding_is_target_residual=True):
        super(Target_Distractor_Module, self).__init__()
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.hidden_size = hidden_size
        self.num_layers = num_conv_layers
        self.is_residual = is_residual
        self.embedding_is_target_residual = embedding_is_target_residual
        self.embedding_layer = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.conv1D_layers = [tf.keras.layers.Conv1D(num_conv_filters, 1, activation='relu') for _ in range(num_conv_layers)]
        self.out_conv_layer = tf.keras.layers.Conv1D(1, 1, activation='relu')

    def call(self, target, distractors):
        """
        Args:
            target: [batch_size, reference_object_size]
            distractors: [batch_size, num_distractors, reference_object_size]
        Returns:
            embedding: [batch_size, hidden_size]
        """
        #Apply first layer to targets and distractor(s), to have them on the shape used for res connections
        embedded_target = self.embedding_layer(target)
        embedded_distractors = self.embedding_layer(distractors)
        target_exp = tf.expand_dims(embedded_target, axis=1)
        inputs = tf.concat([target_exp, embedded_distractors], axis=1)
        for layer in self.conv1D_layers:
            inputs = layer(inputs)
            if self.is_residual:
                inputs = inputs + target
        embedding = self.out_conv_layer(inputs)
        embedding = tf.squeeze(embedding, axis=1)
        if self.embedding_is_target_residual:
            embedding = embedding + target
        return embedding

class Diagonal_Scale_Policy_Head(tf.keras.layers.Layer):
    """
    Policy head that outputs a diagonal covariance matrix
    """
    def __init__(self, hidden_size, num_actions, min_std=1e-6):
        super(Diagonal_Scale_Policy_Head, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.min_std = min_std
        self.mean_layer = tf.keras.layers.Dense(num_actions)
        self.std_layer = tf.keras.layers.Dense(num_actions)
        self.std_activation = lambda x: tf.math.softplus(x) + self.min_std

    def call(self, inputs):
        """
        Args:
            inputs: [batch_size, hidden_size]
        Returns:
            mean: [batch_size, num_actions]
            log_std: [batch_size, num_actions]
        """
        loc = self.mean_layer(inputs)
        scale = self.std_activation(self.log_std_layer(inputs))
        policy = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        return (policy, loc)

class LSTM_residual_Speaker_Agent(tf.keras.models.Model):
    """
    LSTM residual Speaker Agent. Fundamentally a Network with inputs:
    - target [batch_size, target_size]
    - distractors [batch_size, distractor_size] (where distractor_size = target_size)
    - input seq (the already generated sequence), [batch_size, seq_len]
    - language embeddings onto which output residuals are applied [batch_size, seq_len, langauge_embedding_size]
    Returns:
         - policy over the language embedding space, which will be treated as a residual to the output embedding of a LM
    """
    def __init__(self, reference_object_size, num_distractors, vocabulary_size, langauge_embedding_size, hidden_size, num_lstm_layers, td_module_hidden_size, td_module_num_conv_layers, td_module_num_conv_filters, policy_head, td_module_is_residual=True, td_module_embedding_is_target_residual=True, reused_embedding_weights=None):
        super(LSTM_residual_Speaker_Agent, self).__init__()
        #parameters
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.vocabulary_size = vocabulary_size
        self.langauge_embedding_size = langauge_embedding_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.reused_embedding_weights = reused_embedding_weights
        self.td_module_hidden_size = td_module_hidden_size
        self.td_module_num_conv_layers = td_module_num_conv_layers
        self.td_module_num_conv_filters = td_module_num_conv_filters
        self.td_module_is_residual = td_module_is_residual
        self.td_module_embedding_is_target_residual = td_module_embedding_is_target_residual

        #network elements
        self.td_module = Target_Distractor_Module(reference_object_size=self.reference_object_size, num_distractors=self.num_distractors, hidden_size=self.td_module_hidden_size, num_conv_layers=self.td_module_num_conv_layers, num_conv_filters=self.td_module_num_conv_filters, is_residual=self.td_module_is_residual, embedding_is_target_residual=self.td_module_embedding_is_target_residual)
        self.lstm_input_layer = tf.keras.layers.Dense(self.hidden_size, activation='relu') # not strictly necessary but makes sure
        self.lstm_layers = [tf.keras.layers.LSTM(self.hidden_size, return_sequences=True) for _ in range(self.num_lstm_layers)]
        self.embedding_layer = tf.keras.layers.Embedding(self.vocabulary_size, self.langauge_embedding_size)
        self.underlying_model = self._build_underlying_model()
        if self.reused_embedding_weights is not None:
            self.embedding_layer.embeddings.assign(self.reused_embedding_weights)
        self.policy_head = self.build_policy_head(policy_head)



    def _build_underlying_model(self):
        target_placeholder = tf.keras.layers.Input(shape=[None,self.reference_object_size])
        distractors_placeholder = tf.keras.layers.Input(shape=[None,self.num_distractors,self.reference_object_size])
        input_seq_placeholder = tf.keras.layers.Input(shape=[None, None])
        language_embedding_placeholder = tf.keras.layers.Input(shape=[None, None, self.langauge_embedding_size])

        #prepare LSTM input
        target_distractor_embedding = self.td_module(target_placeholder, distractors_placeholder) # [batch_size, td_module_hidden_size]
        target_distractor_embedding_exp = tf.expand_dims(target_distractor_embedding, axis=1) # [batch_size, 1, td_module_hidden_size]
        target_distractor_embedding_seq = tf.tile(target_distractor_embedding_exp, [1, tf.shape(input_seq_placeholder)[1], 1]) # [batch_size, seq_len, td_module_hidden_size]
        token_embeddings = self.embedding_layer(input_seq_placeholder) # [batch_size, seq_len, langauge_embedding_size]
        lstm_input = tf.concat([target_distractor_embedding_seq, token_embeddings, language_embedding_placeholder], axis=2) # [batch_size, seq_len, td_module_hidden_size + token_embedding_size + langauge_embedding_size]
        seq_activation = self.lstm_input_layer(lstm_input) # [batch_size, seq_len, hidden_size]
        for lstm_layer in self.lstm_layers:
            seq_activation = lstm_layer(seq_activation)
        underlying_model = tf.keras.models.Model(inputs=[target_placeholder, distractors_placeholder, input_seq_placeholder, language_embedding_placeholder], outputs=seq_activation)
        # build underlying model by running it on some dummy data
        underlying_model(tf.zeros([1,self.reference_object_size]), tf.zeros([1,self.num_distractors,self.reference_object_size]), tf.zeros([1,1]), tf.zeros([1,1,self.langauge_embedding_size]))
        return underlying_model

    def build_policy_head(self, policy_head):
        policy_head = policy_head(self.hidden_size, self.langauge_embedding_size)
        #run on some dummy data to build the model
        policy_head(tf.zeros([1,1,self.hidden_size]))
        return policy_head

    def call(self, target, distractors, input_seq, language_embedding_seq):
        seq_activation = self.underlying_model([target, distractors, input_seq, language_embedding_seq])
        policy, loc = self.policy_head(seq_activation)
        return policy, loc

        



