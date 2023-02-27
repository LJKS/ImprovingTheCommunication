import LM_Residual_Signalling_Games
import tensorflow as tf
import agents
import time


def td_error(V_s, rewards, gamma=0.99):
    td = rewards + gamma * V_s[1:] - V_s[:-1]
    return td

def gae(td_error, gamma=0.99, lam=0.95):
    #TODO verify this!
    def scan_gae_function(td_t, td_tplus1, lam, gamma):
        return td_t + lam * gamma * td_tplus1
    my_scan = lambda td_t, td_tplus1: scan_gae_function(td_t, td_tplus1, lam, gamma)
    gae = tf.scan(my_scan, td_error, reverse=True, initializer=0.)
    return gae




def sample_runs(signalling_game, target_num_steps):
    signalling_game.reset_all()
    num_steps = 0
    trajectories = []
    type_specs = {
        'speaker_actions': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        'receiver_actions': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'speaker_action_log_probs': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'receiver_log_probs': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'rewards': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'tokens': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'seq_len': tf.TensorSpec(shape=(), dtype=tf.int32),
        'target': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'distractors': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        'target_distractor_tensor': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        'target_distractor_ndarray_target_idxs': tf.TensorSpec(shape=(), dtype=tf.int32),
        'LM_sequence_embeddings': tf.TensorSpec(shape=(None, 768), dtype=tf.float32)}
    while num_steps < target_num_steps:
        resulting_trajectories = signalling_game.step()
        for trajectory in resulting_trajectories:
            trajectory_steps = trajectory['seq_len']
            assert trajectory_steps > 1
            num_steps += trajectory_steps
            trajectory['speaker_actions'] = tf.stack(trajectory['speaker_actions'], axis=0)
            trajectory['receiver_actions'] = tf.stack(trajectory['receiver_actions'], axis=0)
            trajectory['speaker_action_log_probs'] = tf.stack(trajectory['speaker_action_log_probs'], axis=0)
            trajectory['receiver_log_probs'] = tf.stack(trajectory['receiver_predictions'], axis=0)
            trajectory.pop('receiver_predictions')
            trajectory['rewards'] = tf.stack(trajectory['rewards'], axis=0)
            trajectory['tokens'] = tf.stack(trajectory['tokens'], axis=0)
            trajectory['seq_len'] = tf.convert_to_tensor(trajectory['seq_len'], dtype=tf.int32)
            trajectory['target_distractor_ndarray_target_idxs'] = tf.convert_to_tensor(trajectory['target_distractor_ndarray_target_idxs'], dtype=tf.int32)
            assert sorted(list(trajectory.keys())) == sorted(list(type_specs.keys())), "The trajectory keys do not match the type specs keys: Trajectory has: {} vs typespec has:{}".format([item for item in sorted(list(trajectory.keys())) if not item in sorted(list(type_specs.keys()))], [item for item in sorted(list(type_specs.keys())) if not item in sorted(list(trajectory.keys()))])
            for key in type_specs.keys():
                assert type_specs[key].is_compatible_with(trajectory[key])
            trajectories.append(trajectory)
    #create a dataset from the trajectories via a from_generator and provide respective tf.TypeSpecs
    dataset = tf.data.Dataset.from_generator(lambda : trajectories, output_signature=type_specs)
    return dataset

def prepare_ppo_ds(dataset, critic):
    dataset = dataset.map(lambda x: (x['speaker_actions'], x['speaker_action_log_probs'], x['rewards'], x['tokens'], x['seq_len'], x['target'], x['distractors'], x['LM_sequence_embeddings']))
    #compute V(s) for the sequence
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, critic(target, distractors, tokens, LM_sequence_embeddings)))
    #TODO this is for safety, remove later or mark with a safety flag
    for elem in dataset:
        speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, V_s = elem
        assert len(V_s.shape) == 1
        assert V_s[-1] == 0.
        assert len(rewards.shape)==1
        assert rewards.shape[0] == V_s.shape[0] - 1
    # compute the TD error
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, V_s: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, V_s, td_error(V_s, rewards)))
    #compute the GAEs
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, V_s, td_error: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, V_s, td_error, gae(td_error)))




def main():
    # Create a vectorized Environment
    underlying_LM, tokenizer = LM_Residual_Signalling_Games.load_gpt2_TF()
    output_embeddings = underlying_LM.get_output_embeddings()
    vocab_size = tokenizer.vocab_size
    max_length = 150
    num_distractors = 1
    underlying_signalling_game = LM_Residual_Signalling_Games.Underlying_Signalling_Game
    # sgg = signalling_game_generator
    sgg_train, sgg_test, sgg_val = LM_Residual_Signalling_Games.create_coco_caption_signalling_game_data(num_distractors=num_distractors)
    policy_head = agents.Diagonal_Scale_Policy_Head
    speaker_agent = agents.LSTM_residual_Speaker_Agent(reference_object_size=2048, num_distractors=1,
                                                       vocabulary_size=vocab_size, language_embedding_size=768,
                                                       hidden_size=256, num_lstm_layers=2, td_module_hidden_size=2048,
                                                       td_module_num_conv_layers=2, td_module_num_conv_filters=2,
                                                       policy_head=policy_head, td_module_is_residual=True)
    receiver_agent = agents.Receiver_LSTM_Agent(reference_object_size=2048, num_distractors=1,
                                                vocabulary_size=vocab_size, language_embedding_size=768,
                                                hidden_size=256, num_lstm_layers=1, refmod_hidden_size=512,
                                                refmod_num_conv_layers=2, refmod_conv_filters=2)
    critic = agents.Sender_LSTM_Critic(reference_object_size=2048, num_distractors=1,
                                                       vocabulary_size=vocab_size, language_embedding_size=768,
                                                       hidden_size=256, num_lstm_layers=2, td_module_hidden_size=2048,
                                                       td_module_num_conv_layers=2, td_module_num_conv_filters=2, td_module_is_residual=True)
    env = LM_Residual_Signalling_Games.Vectorized_LM_Residual_Signalling_Game(residual_sender_agent=speaker_agent, receiver_agent=receiver_agent,
                                                 underlying_LM=underlying_LM, LM_tokenizer=tokenizer,
                                                 underlying_signalling_game=underlying_signalling_game,
                                                 signalling_game_generator=iter(sgg_train),
                                                 num_distractors=num_distractors, vocab_size=vocab_size,
                                                 max_length=max_length, reward_function=LM_Residual_Signalling_Games.simple_reward_function,
                                                 batch_size=128)
    tt = time.time()
    dataset = sample_runs(env, 10000)
    for i in dataset:
        print(i['tokens'], i['speaker_actions'])
    print(time.time() - tt)
if __name__ == '__main__':
    main()
