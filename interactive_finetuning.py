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
    gae = tf.scan(fn=my_scan, elems=td_error, reverse=True, initializer=0.)
    return gae

#implements the discounted rewards to go, i.e. the MC-estimate of the the state-value
def rewards_to_go(rewards, gamma=0.99):
    discounted_rewards = tf.scan(fn=lambda a, x: x + gamma * a, elems=rewards, reverse=True, initializer=0.)
    return discounted_rewards

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
        'target_idx': tf.TensorSpec(shape=(), dtype=tf.int32),
        'LM_sequence_embeddings': tf.TensorSpec(shape=(None, 768), dtype=tf.float32)}
    while num_steps < target_num_steps:
        resulting_trajectories = signalling_game.step()
        for trajectory in resulting_trajectories:
            trajectory_steps = trajectory['seq_len']
            #assert trajectory_steps > 1
            num_steps += trajectory_steps
            trajectory['speaker_actions'] = tf.stack(trajectory['speaker_actions'], axis=0)
            trajectory['receiver_actions'] = tf.stack(trajectory['receiver_actions'], axis=0)
            trajectory['speaker_action_log_probs'] = tf.stack(trajectory['speaker_action_log_probs'], axis=0)
            trajectory['receiver_log_probs'] = tf.stack(trajectory['receiver_predictions'], axis=0)
            trajectory.pop('receiver_predictions')
            trajectory['rewards'] = tf.stack(trajectory['rewards'], axis=0)
            trajectory['tokens'] = tf.stack(trajectory['tokens'], axis=0)
            trajectory['seq_len'] = tf.convert_to_tensor(trajectory['seq_len'], dtype=tf.int32)
            trajectory['target_idx'] = tf.convert_to_tensor(trajectory['target_distractor_ndarray_target_idxs'], dtype=tf.int32)
            trajectory.pop('target_distractor_ndarray_target_idxs')
            assert sorted(list(trajectory.keys())) == sorted(list(type_specs.keys())), "The trajectory keys do not match the type specs keys: Trajectory has: {} vs typespec has:{}".format([item for item in sorted(list(trajectory.keys())) if not item in sorted(list(type_specs.keys()))], [item for item in sorted(list(type_specs.keys())) if not item in sorted(list(trajectory.keys()))])
            for key in type_specs.keys():
                assert type_specs[key].is_compatible_with(trajectory[key])
            trajectories.append(trajectory)
    #create a dataset from the trajectories via a from_generator and provide respective tf.TypeSpecs
    dataset = tf.data.Dataset.from_generator(lambda : trajectories, output_signature=type_specs)
    return dataset

def prepare_ppo_ds(dataset, critic, gamma=0.99, lam=0.95, value_function_batching=128):
    dataset = dataset.map(lambda x: (x['speaker_actions'], x['speaker_action_log_probs'], x['rewards'], x['tokens'], x['seq_len'], x['target'], x['distractors'], x['LM_sequence_embeddings'], x['target_distractor_tensor'], x['target_idx']))
    #compute V(s) for the sequence, first needs (padded) batching, afterwards unbatching
    dataset = dataset.padded_batch(value_function_batching, padded_shapes=([None, None], [None], [None], [None], [], [None], [None, None], [None, 768], [None, None], []))
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, tf.squeeze(critic(target, distractors, tokens[:,:-1], LM_sequence_embeddings), axis=2)))
    dataset = dataset.unbatch()
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s: (speaker_actions[:seq_len,:], speaker_action_log_probs[:seq_len], rewards[:seq_len], tokens[:seq_len+1], seq_len, target, distractors, LM_sequence_embeddings[:seq_len], target_distractor_tensor, target_idx, V_s[:seq_len]))
    #last V_s is zero by definition (no next state), this is required to define the td-error as r + gamma * V(s') - V(s) [where V(s') is required and should be zero for the last state]
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, tf.concat([V_s, tf.zeros(shape=1, dtype=tf.float32)], axis=0)))
    #TODO this is for safety, remove later or mark with a safety flag
    for elem in dataset:
        speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s = elem
        assert len(V_s.shape) == 1
        assert V_s[-1] == 0.
        assert len(rewards.shape)==1
        assert rewards.shape[0] == V_s.shape[0] - 1
    # compute the TD error
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error(V_s, rewards)))
    #compute the GAEs
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae(td_error)))
    #compute the rewards to go
    dataset = dataset.map(lambda speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae: (speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae, rewards_to_go(rewards)))
    return dataset

def train_sender_ppo(data, sender, sender_optimizer, kl_cutoff=1.5, kl_averaging_weight=0.95, entropy_regularization_factor=0.001, clip_ratio=0.2, num_epochs=10):
    #trains the sender agent via PPO
    #rewrite this for timeseries data!TODO
    def train_step(target, distractors, input_seq, language_embedding_seq, speaker_actions, old_log_probs, advantages, seq_len, clip_ratio=0.2):
        target_mask = tf.sequence_mask(seq_len, maxlen=tf.shape(input_seq)[1]-1, dtype=tf.float32)
        #train step for the sender agent
        with tf.GradientTape() as tape:
            #compute the loss
            #TODO potentially check the validity of applying the target mask here
            sender_policy, _loc = sender(target, distractors, input_seq[:,:-1], language_embedding_seq)
            action_log_probs = sender_policy.log_prob(speaker_actions)
            p_ratio = tf.exp(action_log_probs - old_log_probs)
            advantage_ratio_scores = p_ratio * advantages
            clipped_advantage_ratio_scores = tf.clip_by_value(p_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            ppo_objective = tf.minimum(advantage_ratio_scores, clipped_advantage_ratio_scores) * target_mask
            #print('target_mask', target_mask.shape, 'ppo', ppo_objective.shape, 'seq_len', seq_len, 'input_seq', input_seq.shape, 'old_log_probs', old_log_probs.shape)
            kl_estimate = old_log_probs - action_log_probs
            kl_estimate = tf.reduce_mean(kl_estimate)
            entropy_estimate = tf.reduce_mean(action_log_probs * target_mask)
            # negative because by default TF is minimizing
            ppo_loss = - (tf.reduce_mean(ppo_objective) + entropy_regularization_factor * entropy_estimate)
        print(f'entropy_estimate vs analytic: {tf.reduce_mean(sender_policy.entropy()*target_mask).numpy()} vs {entropy_estimate.numpy()}')
        #compute the gradients
        gradients = tape.gradient(ppo_loss, sender.trainable_variables)
        #apply the gradients
        sender_optimizer.apply_gradients(zip(gradients, sender.trainable_variables))
        #return the loss
        return ppo_loss, kl_estimate, entropy_estimate
    #train the sender agent
    average_kl = tf.zeros(shape=1, dtype=tf.float32)
    for epoch in range(num_epochs):
        if average_kl > kl_cutoff:
            break
        for speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae, rewards_to_go in data:
            ppo_loss, kl_estimate, entropy_estimate = train_step(target=target, distractors=distractors, input_seq=tokens, language_embedding_seq=LM_sequence_embeddings, speaker_actions=speaker_actions, old_log_probs=speaker_action_log_probs, advantages=gae, clip_ratio=clip_ratio, seq_len=seq_len)
            average_kl = kl_averaging_weight * average_kl + (1 - kl_averaging_weight) * kl_estimate
            if average_kl > kl_cutoff:
                print(f'stopping training due to high kl {average_kl.numpy()} > {kl_cutoff}')
                break

def train_critic(critic, data, critic_optimizer, num_epochs=1):
    def squared_error_loss(y_true, y_pred):
        return tf.square(y_true - y_pred)
    def train_step(target, distractors, input_seq, language_embedding_seq, rewards_to_go, seq_len):
        target_mask = tf.sequence_mask(seq_len, maxlen=tf.shape(input_seq)[1]-1, dtype=tf.float32)
        with tf.GradientTape() as tape:
            critic_output = critic(target, distractors, input_seq[:,:-1], language_embedding_seq)
            critic_output = tf.squeeze(critic_output, axis=-1)
            #print('rewards_to_go', rewards_to_go.shape, 'critic_output', critic_output.shape, 'target_mask', target_mask.shape)
            critic_loss = squared_error_loss(rewards_to_go, critic_output)
            #print('critic_loss', critic_loss.shape)
            critic_loss = tf.reduce_mean(critic_loss * target_mask)
            critic_loss = critic_loss / tf.reduce_mean(target_mask)
        #compute the gradients
        gradients = tape.gradient(critic_loss, critic.trainable_variables)
        #apply the gradients
        critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        #return the loss
        return critic_loss

    for epoch in range(num_epochs):
        for speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae, rewards_to_go in data:
            critic_loss = train_step(target=target, distractors=distractors, input_seq=tokens, language_embedding_seq=LM_sequence_embeddings, rewards_to_go=rewards_to_go, seq_len=seq_len)


def train_receiver_supervised(receiver, data, receiver_optimizer, num_epochs=1, num_distractors=1):
    cce = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    def train_step(message, target_distractor_tensor, target_idx, seq_len):
        target_mask = tf.sequence_mask(seq_len, maxlen=tf.shape(message)[1], dtype=tf.float32)
        target_idx = tf.one_hot(target_idx, depth=num_distractors+1)
        # tile targets along the timesteps dimension, from (batch_size, num_distractors+1) to (batch_size, timesteps, num_distractors+1)
        target_idx = tf.tile(tf.expand_dims(target_idx, axis=1), [1, tf.shape(message)[1], 1])
        with tf.GradientTape() as tape:
            receiver_output = receiver(message, target_distractor_tensor)
            cce_loss = cce(target_idx, receiver_output) * target_mask
            cce_loss = tf.reduce_mean(cce_loss) / tf.reduce_mean(target_mask)
        #compute the gradients
        gradients = tape.gradient(cce_loss, receiver.trainable_variables)
        #apply the gradients
        receiver_optimizer.apply_gradients(zip(gradients, receiver.trainable_variables))
        #return the loss
        return cce_loss
    for epoch in range(num_epochs):
        for speaker_actions, speaker_action_log_probs, rewards, tokens, seq_len, target, distractors, LM_sequence_embeddings, target_distractor_tensor, target_idx, V_s, td_error, gae, rewards_to_go in data:
            cce_loss = train_step(message=tokens, target_distractor_tensor=target_distractor_tensor, target_idx=target_idx, seq_len=seq_len)




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
    sender_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    receiver_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for i in range(10):
        dataset = sample_runs(env, 4000)
        print(f'epoch {i}, sampled dataset')
        dataset = prepare_ppo_ds(dataset, critic)
        print(f'epoch {i}, prepared dataset')
        #print('check here')
        #for elem in dataset.take(1):
        #    for e, n in zip(elem, ['speaker_actions', 'speaker_action_log_probs', 'rewards', 'tokens', 'seq_len', 'target', 'distractors', 'LM_sequence_embeddings', 'target_distractor_tensor', 'target_idx', 'V_s', 'td_error', 'gae', 'rewards_to_go']):
        #        print(n, e.shape)
        dataset = dataset.cache().shuffle(4000).padded_batch(32, padded_shapes=(
        [None, None], [None], [None], [None], [], [None], [None, None], [None, 768], [None, None], [], [None], [None], [None], [None]))

        train_sender_ppo(sender=speaker_agent, data=dataset, sender_optimizer=sender_optimizer, num_epochs=1)
        #print(speaker_agent.summary())
        receiver_agent(tf.zeros(shape=(128, 10), dtype=tf.int32), tf.zeros(shape=(128, 2, 2048), dtype=tf.float32))
        #print(receiver_agent.summary())
        print(critic.summary())
        print(f'epoch {i}, trained sender')
        train_receiver_supervised(receiver=receiver_agent, data=dataset, receiver_optimizer=receiver_optimizer)
        print(f'epoch {i}, trained receiver')
        train_critic(critic=critic, data=dataset, critic_optimizer=critic_optimizer)
        print(f'epoch {i}, trained critic')

if __name__ == '__main__':
    main()
