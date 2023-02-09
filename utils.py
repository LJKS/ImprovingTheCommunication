import tensorflow as tf

EOS_TOKEN = " <ENDOFSEQUENCE> "
START_TOKEN = " <STARTOFSEQUENCE> "


def eos_mask(
    eos_token,
    sequence,
    batch_size,
    first_eos_included=tf.zeros(shape=(1,), dtype=tf.bool),
):
    # for a batch of sequences create a mask that covers all elements after the first eos in each sequence with 0, everything else is 1

    sequence = tf.transpose(sequence)
    # first state element represents mask with first occurence of eos excluded, second included
    sequence = tf.scan(
        lambda states, seq_element: (
            tf.math.logical_and(states[0], (seq_element != eos_token)),
            states[0],
        ),
        sequence,
        (tf.ones(batch_size, dtype=tf.bool), tf.ones(batch_size, dtype=tf.bool)),
    )
    sequence = tf.cond(first_eos_included, lambda: sequence[1], lambda: sequence[0])
    sequence = tf.transpose(sequence)
    return sequence
