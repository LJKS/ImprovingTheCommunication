import transformers as hf
import tensorflow as tf
import data
import test_huggingface
import coco_gpt2_datasets
import time
import numpy as np


def main(num_epochs=10):
    train, test, val = coco_gpt2_datasets.load_default_resnetfeature_gpt2tokencaption_stringid_cocodatasets()
    gpt2_model, tokenizer = test_huggingface.load_gpt2_TF()
    #get gpt2 output embeddings
    output_embedding_layer = gpt2_model.get_output_embeddings()
    # get the vocab size
    vocab_size = tokenizer.vocab_size
    #build a functional tf model which maps the 2048 resnet features to the vocab size via a lstm
    input = tf.keras.layers.Input(shape=(2048,))
    input_seq = tf.keras.layers.Input(shape=(None,))
    #input layer
    input_layer = tf.keras.layers.Dense(768)
    #embedding layer
    embedding_layer = tf.keras.layers.Embedding(vocab_size, 768)
    #lstm layer
    lstm_layer1 = tf.keras.layers.LSTM(768, return_sequences=True)
    lstm_layer2 = tf.keras.layers.LSTM(768, return_sequences=True)
    #dense layer
    dense_layer = tf.keras.layers.Dense(768)
    #build sequential model
    img_embedding = input_layer(input)
    #reshape img embedding to (batch_size, 1, 756)
    img_embedding = tf.reshape(img_embedding, (-1, 1, 768))
    #stack image embeddings along seq len dim
    seq_len = tf.shape(input_seq)[1]
    img_embedding = tf.tile(img_embedding, (1, seq_len, 1))
    #get token embeddings
    token_embeddings = embedding_layer(input_seq)
    #concatenate token and image embeddings
    concat_embeddings = tf.concat([img_embedding, token_embeddings], axis=2)
    #pass through lstm
    lstm_output = lstm_layer1(concat_embeddings)
    lstm_output = lstm_layer2(lstm_output)
    #pass through dense layer
    dense_output = dense_layer(lstm_output)
    #build model
    model = tf.keras.Model(inputs=[input, input_seq], outputs=dense_output)
    #run model on dummy data
    dummy_img = tf.zeros((1, 2048))
    dummy_seq = tf.zeros((1, 10))
    model_output = model([dummy_img, dummy_seq])

    #set model embedding weights to gpt2 output embedding weights
    embedding_layer.embeddings.assign(output_embedding_layer.weight)

    for elem in train.take(1):
        img_feature, caption, _resulting_string, _image_id = elem
        print(img_feature)
        print(caption)

    def prepate_ds(ds):
        ds = ds.map(lambda img_feature, caption, _resulting_string, _image_id: (img_feature, tf.cast(caption, tf.int32)))
        ds = ds.map(lambda img_feature, caption: (img_feature, caption[:-1], caption[1:]))
        ds = ds.map(lambda img_feature, caption, target: (img_feature, caption, target, tf.ones_like(caption)))
        ds = ds.cache().shuffle(50000).padded_batch(128, padding_values=(0.,0,0,0))
        ds = ds.map(lambda img_feature, caption, target, pred_mask: (img_feature, caption, target, tf.cast(pred_mask, tf.float32)))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    train = prepate_ds(train)
    test = prepate_ds(test)
    val = prepate_ds(val)
    @tf.function
    def train_step(img, seq, target, gpt_pred, optimizer, model, loss_fn, loss_mask, output_embedding_layer):
        with tf.GradientTape() as tape:
            out_embedding_residuals = model([img, seq])
            output_embedding = out_embedding_residuals + gpt_pred
            pred = tf.nn.softmax(output_embedding_layer(output_embedding))
            loss = loss_fn(target, pred)
            loss = tf.reduce_sum(loss * loss_mask)/tf.reduce_sum(loss_mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        seq_log_prob = -loss_fn(target, pred)
        seq_n_tokens = tf.reduce_sum(loss_mask, axis=-1, keepdims=True)
        token_pp = tf.math.pow(tf.exp(seq_log_prob*loss_mask), 1/seq_n_tokens)
        seq_pp = tf.reduce_prod(token_pp, axis=-1)
        pp = tf.reduce_mean(seq_pp)
        return loss, out_embedding_residuals, gpt_pred, pp

    @tf.function
    def test_step(img, seq, target, gpt_pred, model, loss_fn, loss_mask, output_embedding_layer):
        out_embedding_residuals = model([img, seq])
        output_embedding = out_embedding_residuals + gpt_pred
        pred = tf.nn.softmax(output_embedding_layer(output_embedding))
        loss = loss_fn(target, pred)
        loss = tf.reduce_sum(loss * loss_mask)/tf.reduce_sum(loss_mask)
        seq_log_prob = -loss_fn(target, pred)
        seq_n_tokens = tf.reduce_sum(loss_mask, axis=-1, keepdims=True)
        token_pp = tf.math.pow(tf.exp(seq_log_prob*loss_mask), 1/seq_n_tokens)
        seq_pp = tf.reduce_prod(token_pp, axis=-1)
        pp = tf.reduce_mean(seq_pp)
        return loss, out_embedding_residuals, gpt_pred, pp

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    output_embedding_layer = tf.keras.layers.Dense(vocab_size)
    #run output embedding layer on dummy data with 756 input dim
    dummy_output_embedding = output_embedding_layer(tf.zeros((1,1, 768)))
    #set output embedding layer weights to gpt2 output embedding weights
    output_embedding_layer.set_weights(output_embedding_layer.get_weights())

    for epoch in range(num_epochs):
        losses = []
        pps = []
        for img, seq, target, pred_mask in train.take(10):
            gpt_pred = gpt2_model(seq, return_dict=True, output_hidden_states=True).hidden_states[-1]
            loss, _, _, pp = train_step(img, seq, target, gpt_pred, optimizer, model, loss_fn, pred_mask, output_embedding_layer)
            losses.append(loss.numpy())
            pps.append(pp.numpy())
        print('epoch: ', epoch, 'loss: ', np.mean(losses), 'pp: ', np.mean(pps))

    return model, gpt2_model, tokenizer, [train, test, val], output_embedding_layer, [train_step, test_step]

if __name__ == '__main__':
    main()