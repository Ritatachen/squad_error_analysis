import argparse
import datetime
import os
import typing

import tensorflow as tf

# constant values
CONST = {
    'PAD': 0,
    'START': 1,
    'END': 2,
    'UNK': 3,
}
MAX_LENGTH = 40


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int) -> None:
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        cells = [
            tf.keras.layers.LSTMCell(enc_units),
            tf.keras.layers.LSTMCell(enc_units),
        ]
        self.model = tf.keras.layers.StackedRNNCells(cells, return_sequences=True, return_state=True)

    def call(self, x: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        x = self.embedding(x)
        output = self.model(x)
        print(output)
        return output, None


class BahdanauAttention(tf.keras.Model):

    def __init__(self, units: int) -> None:
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query: tf.Tensor, values: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor) \
            -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def get_index_table_from_file(path: str) -> tf.lookup.StaticHashTable:
    table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER
        ),
        CONST['UNK'] - len(CONST)
    )
    return table


def get_dataset(src_path: str, table: tf.lookup.StaticHashTable) -> tf.data.Dataset:

    def to_ids(text):
        tokenized = tf.strings.split(tf.reshape(text, [1]), sep=' ')
        ids = table.lookup(tokenized.values) + len(CONST)
        return ids

    def add_start_end_tokens(tokens):
        ids = tf.concat([[CONST['START']], tf.cast(tokens, tf.int32), [CONST['END']]], axis=0)
        return ids

    dataset = tf.data.TextLineDataset(src_path)
    dataset = dataset.map(to_ids)
    dataset = dataset.map(add_start_end_tokens)
    return dataset


def filter_instance_by_max_length(src: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
    return tf.logical_and(tf.size(src) <= MAX_LENGTH, tf.size(tgt) <= MAX_LENGTH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='path to dataset')
    parser.add_argument('--epoch', default=10, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='the number of training instances in one forward/backward pass')
    parser.add_argument('--embedding-dim', default=20, type=int,
                        help='dimension of the dense embeddings')
    parser.add_argument('--hidden_dim', default=1024, type=int,
                        help='dimension of the hidden representations')
    parser.add_argument('--shuffle-buffer-size', default=4096, type=int,
                        help='the number of instances that will be buffered when shuffling the dataset')
    parser.add_argument('--device', default=-1, type=int,
                        help='the device ID to use')
    parser.add_argument('--checkpoint', default='./checkpoints', type=str,
                        help='path to checkpoints')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if 0 <= args.device and 0 < len(gpus):
        # Restrict TensorFlow to only use the specified GPU
        tf.config.experimental.set_visible_devices(gpus[args.device], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'[{datetime.datetime.now()}] {len(gpus)} physical GPUs')
        print(f'[{datetime.datetime.now()}] {len(logical_gpus)} logical GPUs')

    print(f'[{datetime.datetime.now()}] Loading the vocabulary...')
    src_table = get_index_table_from_file(os.path.join(args.dataset, 'src_vocab.txt'))
    tgt_table = get_index_table_from_file(os.path.join(args.dataset, 'tgt_vocab.txt'))

    print(f'[{datetime.datetime.now()}] Loading the preprocessed data...')
    src_train = get_dataset(os.path.join(args.dataset, 'src_train.txt'), src_table)
    tgt_train = get_dataset(os.path.join(args.dataset, 'tgt_train.txt'), tgt_table)
    train_dataset = tf.data.Dataset.zip((src_train, tgt_train))
    train_dataset = train_dataset.filter(filter_instance_by_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(args.shuffle_buffer_size)
    train_dataset = train_dataset.padded_batch(
        args.batch_size,
        padded_shapes=([MAX_LENGTH + 2], [MAX_LENGTH + 2]),
        padding_values=(CONST['PAD'], CONST['PAD']),
        drop_remainder=True,
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # create a model
    print(f'[{datetime.datetime.now()}] Creating a seq2seq model...')
    encoder = Encoder(
        src_table.size().numpy() + len(CONST),
        args.embedding_dim,
        args.hidden_dim
    )
    decoder = Decoder(
        tgt_table.size().numpy() + len(CONST),
        args.embedding_dim,
        args.hidden_dim
    )

    # set up the optimizer
    print(f'[{datetime.datetime.now()}] Setting up the optimizer...')
    optimizer = tf.keras.optimizers.Adam()

    # set up the objective function
    print(f'[{datetime.datetime.now()}] Setting up the objective function...')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        loss_ = loss_object(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, CONST['PAD']))
        mask = tf.cast(mask, dtype=loss_.dtype)
        return tf.reduce_mean(loss_ * mask)

    # set up the saver
    checkpoint_prefix = os.path.join(args.checkpoint, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    @tf.function
    def train_step(src: tf.Tensor, tgt: tf.Tensor):
        _, tgt_length = tgt.shape

        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(src)

            dec_hidden = enc_hidden

            for t in range(tgt_length - 1):
                # using teacher forcing
                dec_input = tf.expand_dims(tgt[:, t], 1)

                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(tgt[:, t + 1], predictions)

        batch_loss = loss / tgt_length

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    print(f'[{datetime.datetime.now()}] Started the training.')
    for epoch in range(args.epoch):
        total_loss = 0

        for batch, (src, tgt) in enumerate(train_dataset):
            print(src)
            batch_loss = train_step(src, tgt)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'[{datetime.datetime.now()}] Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

        # saving (checkpoint) the model
        checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'[{datetime.datetime.now()}] Epoch {epoch + 1} Loss {total_loss:.4f}')


if __name__ == '__main__':
    main()

