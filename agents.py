import tensorflow as tf
import tensorflow_probability as tfp

class Target_Distractor_Module(tf.keras.layers.Layer):
    """
    Module that creates an informed sender (See Lazaridou 2016) takes in a target and distractors and returns a single embedding (contrary to the token directly)
    """
    def __init__(self, reference_object_size, num_distractors, hidden_size, num_conv_layers, num_conv_filters, is_residual=True, embedding_is_target_residual=True):
        super(Target_Distractor_Module, self).__init__()
        if is_residual:
            assert num_conv_filters == num_distractors+1
        if embedding_is_target_residual:
            assert hidden_size == reference_object_size
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.hidden_size = hidden_size
        self.num_layers = num_conv_layers
        self.is_residual = is_residual
        self.embedding_is_target_residual = embedding_is_target_residual
        self.embedding_layer = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.conv1D_layers = [tf.keras.layers.Conv1D(filters=num_conv_filters, kernel_size=1, activation='sigmoid', padding='valid') for _ in range(num_conv_layers)]
        self.out_conv_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation=None, padding='valid')
        self._buildmodel()

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
        embedded_distractors = tf.transpose(embedded_distractors, perm=[0, 2, 1]) # [batch_size, hidden_size, num_distractors]
        target_exp = tf.expand_dims(embedded_target, axis=2) # [batch_size, hidden_size, 1]
        inputs = tf.concat([target_exp, embedded_distractors], axis=2) # [batch_size, hidden_size, num_distractors+1]
        for layer in self.conv1D_layers:
            x = layer(inputs) # [batch_size, hidden_size, num_distractors+1]
            if self.is_residual:
                inputs = inputs + x
            else:
                inputs = x
        embedding = self.out_conv_layer(inputs) # [batch_size, hidden_size, 1]
        embedding = tf.squeeze(embedding, axis=2) # [batch_size, hidden_size]
        if self.embedding_is_target_residual:
            embedding = embedding + target
        return embedding

    def _buildmodel(self):
        #Build the model to create weights
        target = tf.keras.Input(shape=[self.reference_object_size])
        distractors = tf.keras.Input(shape=[self.num_distractors, self.reference_object_size])
        self.call(target, distractors)
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
        self.log_std_layer = tf.keras.layers.Dense(num_actions)
        self.std_activation = lambda x: tf.math.softplus(x) + self.min_std
        self._buildmodel()

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

    def _buildmodel(self):
        #Build the model to create weights
        inputs = tf.keras.Input(shape=[self.hidden_size])
        self.call(inputs)

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
    def __init__(self, reference_object_size, num_distractors, vocabulary_size, language_embedding_size, hidden_size, num_lstm_layers, td_module_hidden_size, td_module_num_conv_layers, td_module_num_conv_filters, policy_head, td_module_is_residual=True, td_module_embedding_is_target_residual=True, reused_embedding_weights=None):
        super(LSTM_residual_Speaker_Agent, self).__init__()
        #parameters
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.vocabulary_size = vocabulary_size
        self.langauge_embedding_size = language_embedding_size
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
        target_placeholder = tf.keras.layers.Input(shape=[self.reference_object_size]) # [batch_size, target_size]
        distractors_placeholder = tf.keras.layers.Input(shape=[self.num_distractors,self.reference_object_size]) # [batch_size, num_distractors, distractor_size=distractor_size]
        input_seq_placeholder = tf.keras.layers.Input(shape=[None])
        language_embedding_placeholder = tf.keras.layers.Input(shape=[None, self.langauge_embedding_size])

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
        underlying_model((tf.zeros([1,self.reference_object_size]), tf.zeros([1,self.num_distractors,self.reference_object_size]), tf.zeros([1,1]), tf.zeros([1,1,self.langauge_embedding_size])))
        return underlying_model

    def build_policy_head(self, policy_head):
        policy_head = policy_head(self.hidden_size, self.langauge_embedding_size)
        return policy_head

    def call(self, target, distractors, input_seq, language_embedding_seq):
        seq_activation = self.underlying_model([target, distractors, input_seq, language_embedding_seq])
        policy, loc = self.policy_head(seq_activation)
        return policy, loc

    def act(self, state_dict):
        target = state_dict["targets"]
        distractors = state_dict["distractors"]
        input_seq = state_dict["token_sequences"]
        language_embedding_seq = state_dict["LM_embeddings"]
        current_token_idxs = state_dict["current_token_idxs"]
        actions, log_probs = self._dif_act(target, distractors, input_seq, language_embedding_seq, current_token_idxs)
        return actions, log_probs


    #seperate function for usage inside of tf.function
    def _dif_act(self, target, distractors, input_seq, language_embedding_seq, current_token_idxs):
        policy, loc = self.call(target, distractors, input_seq, language_embedding_seq)
        actions = policy.sample()
        log_probs = policy.log_prob(actions)
        actions = tf.gather(actions, current_token_idxs, batch_dims=1)
        log_probs = tf.gather(log_probs, current_token_idxs, batch_dims=1)
        return actions, log_probs


class Reference_Object_Module(tf.keras.layers.Layer):
    """
    Module that creates the reference object embedding
    """
    def __init__(self, reference_object_size, num_distractors, hidden_size, num_conv_layers, num_conv_filters, is_residual=True):
        super(Reference_Object_Module, self).__init__()
        if is_residual:
            assert num_conv_filters == num_distractors+1
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.hidden_size = hidden_size
        self.num_layers = num_conv_layers
        self.is_residual = is_residual
        self.embedding_layer = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.conv1D_layers = [tf.keras.layers.Conv1D(filters=num_conv_filters, kernel_size=1, activation='sigmoid', padding='valid') for _ in range(num_conv_layers)]
        self.out_conv_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation=None, padding='valid')
        self._buildmodel()

    def call(self, reference_objects):
        """
        Args:
            target: [batch_size, reference_object_size]
            distractors: [batch_size, num_distractors, reference_object_size]
        Returns:
            embedding: [batch_size, hidden_size]
        """
        #Apply first layer to targets and distractor(s), to have them on the shape used for res connections
        embedded_references = self.embedding_layer(reference_objects) # [batch_size, num_distractors+1, hidden_size]
        embedded_references = tf.transpose(embedded_references, perm=[0, 2, 1]) # [batch_size, hidden_size, num_distractors]
        for layer in self.conv1D_layers:
            x = layer(embedded_references) # [batch_size, hidden_size, num_distractors+1]
            if self.is_residual:
                embedded_references = embedded_references + x
            else:
                embedded_references = x
        embedding = self.out_conv_layer(embedded_references) # [batch_size, hidden_size, 1]
        embedding = tf.squeeze(embedding, axis=2) # [batch_size, hidden_size]
        return embedding

    def _buildmodel(self):
        # Build the model to create weights
        reference_objects = tf.keras.Input(shape=[self.num_distractors+1, self.reference_object_size])
        self.call(reference_objects)

class Receiver_LSTM_Agent(tf.keras.Model):
    def __init__(self, reference_object_size, num_distractors, vocabulary_size, language_embedding_size, hidden_size, num_lstm_layers, refmod_hidden_size, refmod_num_conv_layers, refmod_conv_filters, refmod_residual=True, reused_embedding_weights=None):
        super(Receiver_LSTM_Agent, self).__init__()
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.vocabulary_size = vocabulary_size
        self.language_embedding_size = language_embedding_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.refmod_hidden_size = refmod_hidden_size
        self.refmod_num_conv_layers = refmod_num_conv_layers
        self.refmod_conv_filters = refmod_conv_filters
        self.refmod_residual = refmod_residual
        self.reused_embedding_weights = reused_embedding_weights
        self.embedding_layer = tf.keras.layers.Embedding(vocabulary_size, language_embedding_size)
        self.reference_object_module = Reference_Object_Module(reference_object_size, num_distractors, refmod_hidden_size, refmod_num_conv_layers, refmod_conv_filters, is_residual=refmod_residual)
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, return_sequences=True) for _ in range(num_lstm_layers)]
        self.out_layer = tf.keras.layers.Dense(num_distractors+1, activation='softmax')
        self._buildmodel()
        if reused_embedding_weights is not None:
            self.embedding_layer.set_weights(reused_embedding_weights)

    def call(self, sequence, reference_objects):
        """
        Args:
            sequence: [batch_size, sequence_length]
            reference_objects: [batch_size, num_distractors+1, reference_object_size]
        Returns:
            decision_distribution: [batch_size, num_distractors+1]
        """
        embedded_sequence = self.embedding_layer(sequence) # [batch_size, sequence_length, language_embedding_size]
        embedded_reference_objects = self.reference_object_module(reference_objects) # [batch_size, hidden_size]
        embedded_reference_objects = tf.expand_dims(embedded_reference_objects, axis=1) # [batch_size, 1, hidden_size]
        embedded_reference_objects = tf.tile(embedded_reference_objects, [1, tf.shape(embedded_sequence)[1], 1]) # [batch_size, sequence_length, hidden_size]
        embedded_sequence = tf.concat([embedded_sequence, embedded_reference_objects], axis=2) # [batch_size, sequence_length, language_embedding_size+hidden_size]
        for layer in self.lstm_layers:
            embedded_sequence = layer(embedded_sequence)
        decision_distribution = self.out_layer(embedded_sequence) # [batch_size, sequence_length, num_distractors+1]
        return decision_distribution

    def _buildmodel(self):
        # Build the model to create weights
        sequence = tf.keras.Input(shape=[None])
        reference_objects = tf.keras.Input(shape=[self.num_distractors+1, self.reference_object_size])
        self.call(sequence, reference_objects)

    def act(self, observation_dict):
        messages = observation_dict['token_sequences']
        reference_objects = observation_dict['target_distractor_tensor']
        current_idxs = observation_dict['current_token_idxs']
        action, log_prob = self._dif_act(messages, reference_objects, current_idxs)
        return action, log_prob

    def _dif_act(self, messages, reference_objects, current_idxs):
        predictions = self.call(messages, reference_objects)  # [batch_size, sequence_length, num_distractors+1]
        # now extract the predictions at the current token idxs
        prediction_probs = tf.gather(predictions, current_idxs, batch_dims=1)  # [batch_size, num_distractors+1]
        assert prediction_probs.shape == (messages.shape[0], self.num_distractors + 1)
        prediction_distribution = tfp.distributions.Categorical(probs=prediction_probs)
        actions = prediction_distribution.sample()  # [batch_size]
        log_probs = prediction_distribution.log_prob(actions)  # [batch_size]
        return actions, log_probs


#Mostly a clone of the LSTM Sender, but not with a policy head, but rather a critic head
class Sender_LSTM_Critic(tf.keras.models.Model):
    def __init__(self, reference_object_size, num_distractors, vocabulary_size, language_embedding_size, hidden_size,
                 num_lstm_layers, td_module_hidden_size, td_module_num_conv_layers, td_module_num_conv_filters, td_module_is_residual=True, td_module_embedding_is_target_residual=True,
                 reused_embedding_weights=None):
        super(Sender_LSTM_Critic, self).__init__()
        # parameters
        self.reference_object_size = reference_object_size
        self.num_distractors = num_distractors
        self.vocabulary_size = vocabulary_size
        self.langauge_embedding_size = language_embedding_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.reused_embedding_weights = reused_embedding_weights
        self.td_module_hidden_size = td_module_hidden_size
        self.td_module_num_conv_layers = td_module_num_conv_layers
        self.td_module_num_conv_filters = td_module_num_conv_filters
        self.td_module_is_residual = td_module_is_residual
        self.td_module_embedding_is_target_residual = td_module_embedding_is_target_residual

        # network elements
        self.td_module = Target_Distractor_Module(reference_object_size=self.reference_object_size,
                                                  num_distractors=self.num_distractors,
                                                  hidden_size=self.td_module_hidden_size,
                                                  num_conv_layers=self.td_module_num_conv_layers,
                                                  num_conv_filters=self.td_module_num_conv_filters,
                                                  is_residual=self.td_module_is_residual,
                                                  embedding_is_target_residual=self.td_module_embedding_is_target_residual
                                                  )
        self.lstm_input_layer = tf.keras.layers.Dense(self.hidden_size,
                                                      activation='relu')  # not strictly necessary but makes sure
        self.lstm_layers = [tf.keras.layers.LSTM(self.hidden_size, return_sequences=True) for _ in
                            range(self.num_lstm_layers)]
        self.embedding_layer = tf.keras.layers.Embedding(self.vocabulary_size, self.langauge_embedding_size)
        self.out_layer = tf.keras.layers.Dense(1)

        self.underlying_model = self._build_underlying_model()
        if self.reused_embedding_weights is not None:
            self.embedding_layer.embeddings.assign(self.reused_embedding_weights)
    def _build_underlying_model(self):
        target_placeholder = tf.keras.layers.Input(shape=[self.reference_object_size])  # [batch_size, target_size]
        distractors_placeholder = tf.keras.layers.Input(shape=[self.num_distractors, self.reference_object_size])  # [batch_size, num_distractors, distractor_size=distractor_size]
        input_seq_placeholder = tf.keras.layers.Input(shape=[None])
        language_embedding_placeholder = tf.keras.layers.Input(shape=[None, self.langauge_embedding_size])

        # prepare LSTM input
        target_distractor_embedding = self.td_module(target_placeholder,
                                                     distractors_placeholder)  # [batch_size, td_module_hidden_size]
        target_distractor_embedding_exp = tf.expand_dims(target_distractor_embedding,
                                                         axis=1)  # [batch_size, 1, td_module_hidden_size]
        target_distractor_embedding_seq = tf.tile(target_distractor_embedding_exp,
                                                  [1, tf.shape(input_seq_placeholder)[1],
                                                   1])  # [batch_size, seq_len, td_module_hidden_size]
        token_embeddings = self.embedding_layer(input_seq_placeholder)  # [batch_size, seq_len, langauge_embedding_size]
        lstm_input = tf.concat([target_distractor_embedding_seq, token_embeddings, language_embedding_placeholder],
                               axis=2)  # [batch_size, seq_len, td_module_hidden_size + token_embedding_size + langauge_embedding_size]
        seq_activation = self.lstm_input_layer(lstm_input)  # [batch_size, seq_len, hidden_size]
        for lstm_layer in self.lstm_layers:
            seq_activation = lstm_layer(seq_activation)
        seq_activation = self.out_layer(seq_activation)  # [batch_size, seq_len, 1]
        underlying_model = tf.keras.models.Model(
            inputs=[target_placeholder, distractors_placeholder, input_seq_placeholder, language_embedding_placeholder],
            outputs=seq_activation)
        # build underlying model by running it on some dummy data
        underlying_model((tf.zeros([1, self.reference_object_size]),
                          tf.zeros([1, self.num_distractors, self.reference_object_size]), tf.zeros([1, 1]),
                          tf.zeros([1, 1, self.langauge_embedding_size])))
        return underlying_model

    def call(self, target, distractors, input_seq, language_embedding_seq):
        #tf print all the input shapes
        tf.print("target shape: ", tf.shape(target))
        tf.print("distractors shape: ", tf.shape(distractors))
        tf.print("input_seq shape: ", tf.shape(input_seq))
        tf.print("language_embedding_seq shape: ", tf.shape(language_embedding_seq))

        seq_activation = self.underlying_model([target, distractors, input_seq, language_embedding_seq])
        return seq_activation
