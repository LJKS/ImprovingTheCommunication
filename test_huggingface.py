import tensorflow as tf
import transformers as hf


"""A function loading GPT2 from huggingface. It returns the respective TensorFlow model and tokenizer."""
def load_gpt2_TF():
    model = hf.TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = hf.GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def main():
    model, tokenizer = load_gpt2_TF()
    output_embedding_layer = model.get_output_embeddings()
    print(output_embedding_layer)
    #check the model on the sample sentence "Hello, my cat is cute"
    with tf.GradientTape() as tape:
        input_ids = tf.constant(tokenizer.encode("<|endoftext|> This is bullshit!", add_special_tokens=True))
        outputs = model(input_ids, return_dict=True, output_hidden_states=True)
        print(type(outputs))
        last_hidden_states = outputs.hidden_states[-1]
    output_embedding_layer = model.get_output_embeddings()
    print(output_embedding_layer)
    logits = outputs.logits
    manual_logits = tf.matmul(last_hidden_states, output_embedding_layer.weight, transpose_b=True)
    assert tf.reduce_all(tf.math.equal(logits, manual_logits))
    print(tf.math.equal(logits, manual_logits))

if __name__ == '__main__':
    main()

