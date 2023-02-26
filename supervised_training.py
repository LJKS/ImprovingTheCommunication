import tensorflow as tf
import tqdm
import numpy as np
import coco_gpt2_datasets
import agents
import time

def prepare_dataset(dataset, batch_size, num_distractors, caption_padding_idx=50256, shuffle_buffer_size=10000):
    def data_to_signalling_game(ds):
        ds = ds.map(lambda img_feature, caption_tokens, caption_string, id, gpt2_hidden_state, mask: (img_feature, caption_tokens, gpt2_hidden_state, mask)).cache()
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.padded_batch(num_distractors+1, padded_shapes=([2048], [None], [None, 768], [None]), padding_values=(0., caption_padding_idx, 0., 0.))
        ds = ds.map(lambda img_feat, caption_idxs, gpt2_hidden_state, mask: (img_feat[0,:], img_feat[1:,:], caption_idxs[0,:], gpt2_hidden_state[0,:-1,:], mask[0, :-1]))
        ds = ds.padded_batch(batch_size, padded_shapes=([2048], [num_distractors, 2048], [None], [None, 768], [None]), padding_values=(0., 0., caption_padding_idx, 0., 0.))
        ds = ds.prefetch(20)
        return ds
    return data_to_signalling_game(dataset)

def supervised_pretraining_sender(sender, train_ds, test_ds, decoding_embedding, entropy_factor, optimizer, num_epochs, batch_size, caption_padding=50256):
    """Supervised pretraining of the sender.

        Args:
            sender: The sender model.
            language_model: The language model.
            dataset: The dataset.
            decoding_embedding: The decoding embedding.
            entropy_factor: The entropy factor.
            optimizer: The optimizer.
            num_epochs: The number of epochs.
        """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def train_step(target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask):
        with tf.GradientTape() as tape:
            # Sender predict action and log_props
            policy, loc = sender(target_reference, distractor_references, target_sequence[:,:-1], language_embedding_seq)
            # sample res_action
            res_action = policy.sample()
            log_probs = policy.log_prob(res_action)
            # Language model predict next word
            total_embedding = res_action + language_embedding_seq
            logits = total_embedding@tf.transpose(decoding_embedding)
            # Calculate loss
            loss = loss_fn(target_sequence[:,1:], logits, loss_mask)
            loss_pred = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask) # average loss, exclude padding
            loss_entropy = tf.reduce_sum(-log_probs*loss_mask) / tf.reduce_sum(loss_mask) # average entropy, exclude padding
            loss_total = loss_pred - entropy_factor * loss_entropy
        # Backpropagation
        grads = tape.gradient(loss_total, sender.trainable_variables)
        optimizer.apply_gradients(zip(grads, sender.trainable_variables))
        return loss_total, loss_pred, loss_entropy

    @tf.function
    def test_step(target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask):
        # Sender predict action and log_props
        policy, loc = sender(target_reference, distractor_references, target_sequence[:,:-1], language_embedding_seq)
        # sample res_action
        res_action = policy.sample()
        log_probs = policy.log_prob(res_action)
        # Language model predict next word
        total_embedding = res_action + language_embedding_seq
        logits = total_embedding@tf.transpose(decoding_embedding)
        # Calculate loss
        loss = loss_fn(target_sequence[:,1:], logits, loss_mask)
        loss_pred = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask) # average loss, exclude padding
        loss_entropy = tf.reduce_sum(-log_probs*loss_mask) / tf.reduce_sum(loss_mask) # average entropy, exclude padding
        loss_total = loss_pred - entropy_factor * loss_entropy
        return loss_total, loss_pred, loss_entropy

    with tqdm.trange(num_epochs) as t:
        for epoch in t:
            t.set_description(f"Epoch {epoch}")
            train_losses_total = tf.keras.metrics.Mean()
            train_losses_pred = tf.keras.metrics.Mean()
            train_losses_entropy = tf.keras.metrics.Mean()
            test_losses_total = tf.keras.metrics.Mean()
            test_losses_pred = tf.keras.metrics.Mean()
            test_losses_entropy = tf.keras.metrics.Mean()
            with tqdm.tqdm(enumerate(train_ds), total=400000/batch_size, position=0, leave=True) as train_progress:
                tt = time.time()
                for step, (target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask) in train_progress:
                    print(f'loading took {time.time()-tt} seconds')
                    tt = time.time()
                    loss_total, loss_pred, loss_entropy = train_step(target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask)
                    print(f'trainstep took {time.time()-tt} seconds')
                    tt = time.time()
                    train_losses_total.update_state(loss_total)
                    train_losses_pred.update_state(loss_pred)
                    train_losses_entropy.update_state(loss_entropy)
            for step, (target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask) in enumerate(test_ds):
                loss_total, loss_pred, loss_entropy = test_step(target_reference, distractor_references, target_sequence, language_embedding_seq, loss_mask)
                test_losses_total.update_state(loss_total)
                test_losses_pred.update_state(loss_pred)
                test_losses_entropy.update_state(loss_entropy)
            t.set_postfix_str(f'Total Losses (Train/Test:) {np.mean(train_losses_total):2.4f}/{np.mean(test_losses_total):2.4f} | Prediction Losses (Train/Test:) {np.mean(train_losses_pred):2.4f}/{np.mean(test_losses_pred):2.4f} | Entropy Losses (Train/Test:) {np.mean(train_losses_entropy):2.4f}/{np.mean(test_losses_entropy):2.4f}')

def main():
    # Hyperparameters
    num_epochs = 1
    batch_size = 32
    entropy_factor = 0.001
    learning_rate = 1e-4
    # Load dataset
    train, test, val = coco_gpt2_datasets.load_default_resnetfeature_gpt2tokencaption_gpt2encodings_stringid_cocodatasets()
    train_ds = prepare_dataset(train, batch_size, num_distractors=1)
    test_ds = prepare_dataset(test, batch_size, num_distractors=1)
    gpt2, gpt2_tokenizer = coco_gpt2_datasets.load_gpt2_TF()
    decoding_embedding = gpt2.get_output_embeddings().weight
    # Create sender
    policy_head = agents.Diagonal_Scale_Policy_Head
    vocab_size = gpt2_tokenizer.vocab_size
    sender = agents.LSTM_residual_Speaker_Agent(reference_object_size=2048, num_distractors=1, vocabulary_size=vocab_size,
                                       language_embedding_size=768, hidden_size=256, num_lstm_layers=2,
                                       td_module_hidden_size=2048, td_module_num_conv_layers=2,
                                       td_module_num_conv_filters=2, policy_head=policy_head,
                                       td_module_is_residual=True, reused_embedding_weights=decoding_embedding)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Train
    supervised_pretraining_sender(sender=sender, train_ds=train_ds, test_ds=test_ds, decoding_embedding=decoding_embedding, entropy_factor=entropy_factor, optimizer=optimizer, num_epochs=num_epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()