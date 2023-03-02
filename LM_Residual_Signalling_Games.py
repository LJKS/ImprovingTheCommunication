import tensorflow as tf
import transformers as hf
import gymnasium as gym
import numpy as np
import coco_gpt2_datasets
import agents
import time
def create_coco_caption_signalling_game_data(num_distractors=1, shuffle_buffer_size=10000):
    train, test, val = coco_gpt2_datasets.load_default_resnetfeature_gpt2tokencaption_stringid_cocodatasets()
    def data_to_signalling_game(ds):
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.padded_batch(num_distractors+1, padded_shapes=([2048], [None], [], []), padding_values=(0., 0, "", 0))
        ds = ds.map(lambda img_feat, caption_gpt_idxs, caption, id: (img_feat[0,:], img_feat[1:,:], ((caption_gpt_idxs[0], caption[0], id[0]),(caption_gpt_idxs[1:], caption[1:], id[1:]))))
        return ds
    return data_to_signalling_game(train), data_to_signalling_game(test), data_to_signalling_game(val)




"""
Structure:
 - Signalling_Base_Game: An environment for a signalling game
 - Signalling_Game_Generator: A generator/Dataset yielding tuples of target, disctractor(s) pairs, info object
 - LM_Residual_Signalling_Game_Gym: A class creating a signalling game gym environment for a given LM and based on an underlying signalling game, which is vectorized
"""
def load_gpt2_TF():
    model = hf.TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = hf.GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

"""
A gym like signalling game environment
"""
class Underlying_Signalling_Game():
    def __init__(self, signalling_game_generator, num_distractors, vocab_size, max_length):
        #Underlying attributes
        self.signalling_game_generator = signalling_game_generator
        self.num_distractors = num_distractors
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.number_of_elements = num_distractors + 1

        #Attributes changed during runtime
        self._target = None
        self._target_idx = None
        self._distractors = None
        self._target_distractor_ndarray = None
        self._info = None


    def reset(self):
        #Reset the underlying signalling game, update the target and distractors
        self._target, self._distractors, self._info = next(self.signalling_game_generator)
        target_distractor_list = [self._target] + [self._distractors[i] for i in range(self.num_distractors)] #list of target and distractors
        random_permutation = np.random.permutation(self.number_of_elements)
        self._target_idx = np.squeeze(np.argwhere(random_permutation == 0))
        self._target_distractor_ndarray = tf.stack([tf.squeeze(target_distractor_list[i]) for i in random_permutation])
        #assert shapes are correct
        assert self._target_distractor_ndarray.shape == self.number_of_elements + self._target.shape

    def get_observation_dict(self):
        state_dict = {}
        state_dict["target"] = self._target
        state_dict["distractors"] = self._distractors
        state_dict["target_distractor_ndarray"] = self._target_distractor_ndarray
        state_dict["target_idx"] = self._target_idx
        state_dict["info"] = self._info
        return state_dict


"""
The Vectorized_Signalling_Game_Gym is a gym like environment, which takes a LM and an underlying signalling game and creates a gym environment for the LM residual signalling game
"""
class Vectorized_LM_Residual_Signalling_Game:
    def __init__(self, residual_sender_agent, receiver_agent, underlying_LM, LM_tokenizer, underlying_signalling_game, signalling_game_generator, num_distractors, vocab_size, max_length, reward_function, batch_size=32, bos_token="<|endoftext|>", eos_token="<|endoftext|>"):
        #Underlying attributes
        """
        :param residual_sender_agent:
        :param receiver_agent:
        :param underlying_LM:
        :param underlying_signalling_game:
        :param signalling_game_generator:
        :param num_distractors:
        :param vocab_size:
        :param max_length:
        :param batch_size:
        :param bos_token:
        :param eos_token:
        """
        self.residual_sender_agent = residual_sender_agent
        self.receiver_agent = receiver_agent
        self.underlying_LM = underlying_LM
        self.output_embedding = lambda x: x@ self.underlying_LM.get_output_embeddings().weight.T
        self.LM_tokenizer = LM_tokenizer
        self.underlying_signalling_game_class = underlying_signalling_game
        self.signalling_game_generator = signalling_game_generator
        self.num_distractors = num_distractors
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.reward_function = reward_function
        self.batch_size = batch_size
        self.underlying_signalling_games = [self.underlying_signalling_game_class(self.signalling_game_generator, self.num_distractors, self.vocab_size, self.max_length) for _ in range(self.batch_size)]
        for underlying_signalling_game in self.underlying_signalling_games:
            underlying_signalling_game.reset()
        self.bos_token = self.LM_tokenizer.encode(bos_token)[0]
        self.eos_token = self.LM_tokenizer.encode(eos_token)[0]

        #Attributes changed during runtime
        self._tokens = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of tokens for one batch element
        self._current_token_batch = tf.ragged.constant(self._tokens, dtype=tf.int64).to_tensor(default_value=self.eos_token, shape=[self.batch_size, self.max_length]) #The tokens the agents are currently predicting, outer shape: [batch_size, max_length]
        self._current_token_batch_idxs = np.zeros((self.batch_size,), dtype='i') #The indices of the tokens the agents are currently predicting, shape: [batch_size]
        self._LM_sequence_embeddings = None #The LM embeddings of the tokens the agents are currently predicting, shape: [batch_size, (up to) max_length, embedding_size]
        self._current_lm_embeddings_current_idxs = None #The LM embeddings of the tokens the agents are currently predicting, shape: [batch_size, embedding_size]
        self._speaker_actions = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of speaker actions for one batch element
        self.speaker_action_log_probs = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of speaker action log probs for one batch element
        self._receiver_actions = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of receiver actions for one batch element
        self._receiver_predictions = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of receiver predictions for one batch elelement at one timestep, i.e. of shape [num_distractors + 1]
        self._rewards = [[] for _ in range(self.batch_size)] # list of lists, each list is the sequence of receiver rewards for one batch element
        self._episode_is_finished = [False for _ in range(self.batch_size)] # list of booleans, each boolean is True if the episode is finished for one batch element
        #initial reset
        for i in range(self.batch_size):
            self._reset_and_summarize_seq(i, starting_reset=True)

    def reset_all(self):
        for i in range(self.batch_size):
            self._reset_and_summarize_seq(i, starting_reset=True)
    def _reset_and_summarize_seq(self, i, starting_reset=False):
        """
        Resets the environment and returns the finished sequence
        :param i: the index of the environment to reset
        :return: the finished sequence as a dict
        """
        res_dict = {}
        #WARNING: This does not update the _current_token_batch, which is recomputed centrally in the _reset_and_export_finished_sequences function
        #WARNING: This is not updating the _current_lm_embeddings_current_idxs, which is recomputed centrally in the step function, via _update_lm_embeddings
        #WARNING: This is not updating the _LM_sequence_embeddings, which is recomputed centrally in the step function, via _update_lm_embeddings
        if not starting_reset:
            res_dict['LM_sequence_embeddings'] = self._LM_sequence_embeddings[i][:self._current_token_batch_idxs[i]]

        res_dict["tokens"] = self._tokens[i]
        self._tokens[i] = [self.bos_token]
        res_dict['seq_len'] = self._current_token_batch_idxs[i]
        self._current_token_batch_idxs[i] = 0
        res_dict['speaker_actions'] = self._speaker_actions[i]
        self._speaker_actions[i] = []
        res_dict['speaker_action_log_probs'] = self.speaker_action_log_probs[i]
        self.speaker_action_log_probs[i] = []
        res_dict['receiver_actions'] = self._receiver_actions[i]
        self._receiver_actions[i] = []
        res_dict['receiver_predictions'] = self._receiver_predictions[i]
        self._receiver_predictions[i] = []
        res_dict['rewards'] = self._rewards[i]
        self._rewards[i] = []
        #Don't need this for the return dict, but need to reset it
        self._episode_is_finished[i] = False
        #Reset and gather from the underlying signalling game
        target = self.underlying_signalling_games[i]._target
        res_dict["target"] = target
        distractors = self.underlying_signalling_games[i]._distractors
        res_dict["distractors"] = distractors
        target_distractor_ndarray = self.underlying_signalling_games[i]._target_distractor_ndarray
        res_dict["target_distractor_tensor"] = target_distractor_ndarray
        target_distractor_ndarray_idxs = self.underlying_signalling_games[i]._target_idx
        res_dict["target_distractor_ndarray_target_idxs"] = target_distractor_ndarray_idxs
        self.underlying_signalling_games[i].reset()

        return res_dict



    def step(self):
        """
        Computes one step through all environments and agents. Returns the finished sequences after resetting them.
        :return: list of
        """
        #update the LM embeddings, sets self._current_lm_embeddings_current_idxs
        #Notice: This also takes over partial resetting duties (for _LM_sequence_embeddings, _current_lm_embeddings_current_idxs)
        self._update_lm_embeddings()
        #run speaker agents
        speaker_residual_actions, speaker_action_log_probs = self._speaker_act()
        #combine speaker outputs with LM prediction
        self._decode_residual_actions_to_tokens(speaker_residual_actions) #this is where the magic happens, i.e. here the residuals are implemented!
        #run receiver agents
        self._receiver_act()
        # merge reward and finish function in one, as both need the information which pieces are finished and what reward would be given for that
        self._reward_and_finish_sequences()
        # check which are done and need to be exported and reset
        finished_sequences = self._reset_and_export_finished_sequences()
        return finished_sequences



    def _reset_and_export_finished_sequences(self):
        finished_sequences = []
        for i in range(self.batch_size):
            if self._episode_is_finished[i]:
                finished_sequences.append(self._reset_and_summarize_seq(i))
        #update the current token batch, as some might have been reset
        self._current_token_batch = tf.ragged.constant(self._tokens).to_tensor(default_value=self.eos_token, shape=[self.batch_size, self.max_length])
        return finished_sequences

    def _update_lm_embeddings(self):
        lm_embeddings = self.underlying_LM(self._current_token_batch, return_dict=True, output_hidden_states=True).hidden_states[-1]
        self._LM_sequence_embeddings = lm_embeddings
        self._current_lm_embeddings_current_idxs = tf.gather(lm_embeddings, self._current_token_batch_idxs, batch_dims=1)

    def _speaker_act(self):
        speaker_observations = self._get_speaker_observations()
        speaker_residual_actions, speaker_action_probabilities = self.residual_sender_agent.act(speaker_observations)
        for i in range(self.batch_size):
            self._speaker_actions[i].append(speaker_residual_actions[i])
            self.speaker_action_log_probs[i].append(speaker_action_probabilities[i])
        return speaker_residual_actions, speaker_action_probabilities

    def _receiver_act(self):
        receiver_observations = self._get_receiver_observations()
        receiver_actions, receiver_action_probabilities = self.receiver_agent.act(receiver_observations)
        for i in range(self.batch_size):
            self._receiver_actions[i].append(receiver_actions[i].numpy())
            self._receiver_predictions[i].append(receiver_action_probabilities[i].numpy())

    def _decode_residual_actions_to_tokens(self, speaker_residual_actions, temperature=1.0):
        """
        Decodes the residual actions to tokens and appends them to the current token batch.

        """
        action_embedding = self._current_lm_embeddings_current_idxs + speaker_residual_actions
        logits = action_embedding @ tf.transpose(self.underlying_LM.get_input_embeddings().weight)
        logits = logits / temperature
        print(f'average prob of eos token, should be {1/self.vocab_size}, is {tf.reduce_mean(tf.nn.softmax(logits, axis=1)[:,self.eos_token]).numpy()}')
        sampled_tokens = tf.squeeze(tf.random.categorical(logits, 1))
        for i in range(self.batch_size):
            self._tokens[i].append(sampled_tokens[i].numpy())
        ragged_tokens = tf.ragged.constant(self._tokens)
        self._current_token_batch = ragged_tokens.to_tensor(default_value=self.eos_token, shape=[self.batch_size, self.max_length])
        self._current_token_batch_idxs = self._current_token_batch_idxs + 1

    def _get_speaker_observations(self):
        """
        Returns an observation dict with the following keys:
         - 'token_sequences': A batch of sequences, where each sequence is a list of token ids, padded with eos tokens
         - 'LM_embeddings': The embeddings of the sequences, as predicted by the LM, to which the speaker actions are applied as residuals
         - 'targets': The targets of the underlying signalling game
         - 'distractors': The distractors of the underlying signalling game
         - 'target_distractor_ndarray': The target_distractor_ndarray of the underlying signalling game
         - 'target_distractor_ndarray_idxs': The target_distractor_ndarray_idxs, i.e. idx of the target of the underlying signalling game
        :return: dict as specified above
        """
        state_dict = {}
        state_dict["token_sequences"] = self._current_token_batch
        state_dict["current_token_idxs"] = self._current_token_batch_idxs
        state_dict["LM_embeddings"] = self._LM_sequence_embeddings
        targets = [underlying_signalling_game._target for underlying_signalling_game in self.underlying_signalling_games]
        targets = tf.stack(targets)
        state_dict["targets"] = targets
        distractors = [underlying_signalling_game._distractors for underlying_signalling_game in self.underlying_signalling_games]
        distractors = tf.stack(distractors)
        state_dict["distractors"] = distractors
        target_distractor_ndarray = [underlying_signalling_game._target_distractor_ndarray for underlying_signalling_game in self.underlying_signalling_games]
        target_distractor_tensor = tf.stack(target_distractor_ndarray)
        state_dict["target_distractor_tensor"] = target_distractor_tensor
        target_distractor_ndarray_idxs = [underlying_signalling_game._target_idx for underlying_signalling_game in self.underlying_signalling_games]
        target_distractor_ndarray_idxs = tf.stack(target_distractor_ndarray_idxs)
        state_dict["target_distractor_ndarray_idxs"] = target_distractor_ndarray_idxs
        return state_dict

    def _get_receiver_observations(self):
        """
        Returns an observation dict with the following keys:
            - 'token_sequences': A batch of sequences, where each sequence is a list of token ids, padded with eos tokens
            - 'current_token_idxs': The current token idxs of the sequences (i.e. the respective idx of the last token already generated in the sequence, before
            - 'target_distractor_ndarray': The target_distractor_ndarray of the underlying signalling game
        :return: a dict with the aforementioned keys
        """
        state_dict = {}
        state_dict["token_sequences"] = self._current_token_batch
        state_dict["current_token_idxs"] = self._current_token_batch_idxs
        target_distractor_ndarray = [underlying_signalling_game._target_distractor_ndarray for
                                     underlying_signalling_game in self.underlying_signalling_games]
        target_distractor_tensor = tf.stack(target_distractor_ndarray)
        state_dict["target_distractor_tensor"] = target_distractor_tensor
        return state_dict

    def _sequence_is_finished(self, sequence):
        """
        Checks whether a sequence is finished, i.e. whether it ends with eos token while length is >1, or if max_len is reached
        :param sequence: list of tokens
        :return: boolean
        """
        is_finished = (len(sequence) > 1 and sequence[-1] == self.eos_token) or len(sequence) >= self.max_length
        return is_finished

    def _compute_reward(self, message, target_idx, prediction_idx, prediction_probabilities, is_finished):
        """
        Computes the reward for the given message, target_idx, prediction_probabilities and is_finished
        :param message: The message that was sent
        :param target_idx: The target idx of the underlying signalling game
        :param prediction_idx: The prediction idx of the receiver agent
        :param prediction_probabilities: The prediction probabilities of the receiver agent
        :param is_finished: A boolean indicating whether the message is finished or not
        :return: The reward
        """
        reward_input_dict = {}
        reward_input_dict["message"] = message
        reward_input_dict["target_idx"] = target_idx.item()
        reward_input_dict["prediction_idx"] = prediction_idx.item()
        reward_input_dict["prediction_probabilities"] = prediction_probabilities
        reward_input_dict["is_finished"] = is_finished
        reward = self.reward_function(reward_input_dict)
        return reward


    def _reward_and_finish_sequences(self):
        """
        Computes the rewards and checks whether the sequences are finished, sets respective _episode_is_finished and _rewards
        :return: None
        """
        for i in range(self.batch_size):
            sequence_finished = self._sequence_is_finished(self._tokens[i])
            self._episode_is_finished[i] = sequence_finished
            sequence_idx = self._current_token_batch_idxs[i] - 1 #actions are one step behind the token! (first step there is 1 token, zero actions, second step 2 tokens, 1 action, etc)
            sequence_idx = np.squeeze(sequence_idx).item()
            reward = self._compute_reward(self._tokens[i], self.underlying_signalling_games[i]._target_idx, self._receiver_actions[i][sequence_idx], self._receiver_predictions[i][sequence_idx], sequence_finished)
            self._rewards[i].append(reward)


def simple_reward_function(reward_f_dict: dict):
    assert type(reward_f_dict['target_idx']) == type(reward_f_dict['prediction_idx'])
    assert np.asarray(reward_f_dict['target_idx']).shape == reward_f_dict['prediction_probabilities'].shape
    if reward_f_dict['is_finished']:
        if reward_f_dict['target_idx'] == reward_f_dict['prediction_idx']:
            reward = 1.
        else:
            reward = 0.
    else:
        reward = 0.
    return reward


def main():
    # Create a vectorized Environment
    underlying_LM, tokenizer = load_gpt2_TF()
    output_embeddings = underlying_LM.get_output_embeddings()
    vocab_size = tokenizer.vocab_size
    max_length = 150
    num_distractors = 1
    underlying_signalling_game = Underlying_Signalling_Game
    #sgg = signalling_game_generator
    sgg_train, sgg_test, sgg_val = create_coco_caption_signalling_game_data(num_distractors=num_distractors)
    policy_head = agents.Diagonal_Scale_Policy_Head
    speaker_agent = agents.LSTM_residual_Speaker_Agent(reference_object_size=2048, num_distractors=1, vocabulary_size=vocab_size, language_embedding_size=768, hidden_size=256, num_lstm_layers=2, td_module_hidden_size=2048, td_module_num_conv_layers=2, td_module_num_conv_filters=2, policy_head=policy_head, td_module_is_residual=True)
    receiver_agent = agents.Receiver_LSTM_Agent(reference_object_size=2048, num_distractors=1, vocabulary_size=vocab_size, language_embedding_size=768, hidden_size=256, num_lstm_layers=1, refmod_hidden_size=512, refmod_num_conv_layers=2, refmod_conv_filters=2)
    env = Vectorized_LM_Residual_Signalling_Game(residual_sender_agent=speaker_agent, receiver_agent=receiver_agent, underlying_LM=underlying_LM, LM_tokenizer=tokenizer, underlying_signalling_game=underlying_signalling_game, signalling_game_generator=iter(sgg_train), num_distractors=num_distractors, vocab_size=vocab_size, max_length=max_length, reward_function=simple_reward_function, batch_size=128)
    tt = time.time()
    steps = 400
    for _ in range(steps):
        print(len(env.step()))
    print(steps/(time.time() - tt))

if __name__ == "__main__":
    main()