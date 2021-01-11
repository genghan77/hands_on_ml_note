# NLP with RNN and Attention

* Text generation
  
  * Stateless RNN

  ```python
  """
  Creating dataset
  """ 
  tokenizer = keras.preprocessing.text.Tokenizer(char_level=True) # set char_level = True to get the character-level encoding rather than the default word-level encoding
  # The tokenizer converts the text to lowercase by default, you can set it otherwise by lower=False.

  tokenizer.fit_on_texts([shakespeare_text])

  max_id = len(tokenizer.word_index) # number of distinct characters 
  
  dataset_size = tokenizer.document_count # total number of characters

  [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

  """
  Spliting sequential dataset
  """
  train_size = dataset_size * 90 // 100
  dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

  """
  chopping the sequential dataset into multiple windows
  """
  n_steps = 100
  window_length = n_steps + 1 # target = input shifted 1 character ahead 
  dataset = dataset.window(window_length, shift=1, drop_remainder=True)
  # Don't make n_steps too small, as it would make the RNN not able to learn any pattern longer than n_steps
  # By default, window() creates nonoverlapping windows, but to ge tht elargest possile training set, make shift=1. To ensure all windows are exactly 101 characters long, set drop_remainder=True

  # window() method creates nested datset, so need to flat it. 
  dataset = dataset.flat_map(lambda window: window.batch(window_length))
  # example, if you pass the function lambda ds: ds.batch(2) to flat_map(), it will transform the nested datset {{1, 2}, {3, 4, 5, 6}} into {[1,2], [3,4], [5,6]} dataset of tensors of size 2

  batch_size = 32
  dataset = dataset.shuffle(10000).batch(batch_size)
  dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

  dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

  dataset = dataset.prefetch(1)

  """
  building the model
  """
  model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
        dropout=0.2, recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    ])
    history = model.fit(dataset, epochs=20)

  """
  using the model
  """
  def preprocess(texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1 return tf.one_hot(X, max_id)
  X_new = preprocess(["How are yo"])
  Y_pred = model.predict_classes(X_new)
  tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]

  # make use of temperature to generate more diverse and interesting text.

  def next_char(text, temperature=1):
        X_new = preprocess([text])
        y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1 return tokenizer.sequences_to_texts(char_id.numpy())[0]
  ```

  * Stateful RNN
    
    Stateful RNN preserve the final state after processing one training batch and use it as the initial state for the next training batch. So it only make sense if each input sequence in a batch starts exactly where the corresponding sequence in the previous batch left off. 


  ```python
  """
  dataset
  """
  dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

  dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True) # because of the statefulness, so must use shift = n_steps 
  # batch can't be other numbers other than 1

  dataset = dataset.flat_map(lambda window: window.batch(window_length)) 

  dataset = dataset.batch(1)
  dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:])) 
  dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch)) dataset = dataset.prefetch(1)

  tf.train.Dataset.zip(datasets).map(lambda *windows: tf.stack(windows))

  """
  model and training 
  """
  model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, stateful=True,
        dropout=0.2, recurrent_dropout=0.2,
        batch_input_shape=[batch_size, None, max_id]),
        keras.layers.GRU(128, return_sequences=True, stateful=True,
        dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation="softmax"))]) # note that stateful=True

  class ResetStatesCallback(keras.callbacks. Callback): def on_epoch_begin(self, epoch, logs):
    self.model.reset_states() 
  # at the end need to reset the states before go back to the beginning of the text
  
  # after the stateful model is trained, it will only be possible to make predictions for batches of the same size as were used during training. TO avoid this restriction, create an idential stateless model, and copy the stateful model's weights over to this model. 
  ```

* Sentiment Analysis
  
  ```python
  # simple encode
  word_index = keras.datasets.imdb.get_word_index()

  id_to_word = {id_ + 3: word for word, id_ in word_index.items()} >>> for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token

  " ".join([id_to_word[id_] for id_ in X_train[0][:10]]) # to visualize a review

  # preprocess

  def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300) # truncating the reviews, this will speed up traiing and it won't impact performance too much. 
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ") X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ") X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

  # constructing vocab
  from collections import Counter
  vocabulary = Counter()
  for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch: vocabulary.update(list(review.numpy()))
  
  # truncate the vocab
  vocab_size = 10000
  truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]
 
  # create lookup table
  words = tf.constant(truncated_vocabulary)
  word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
  vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
  num_oov_buckets = 1000
  table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


  def encode_words(X_batch, y_batch): return table.lookup(X_batch), y_batch
    train_set = datasets["train"].batch(32).map(preprocess)
    train_set = train_set.map(encode_words).prefetch(1)


   embed_size = 128
   model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                               input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=5)

    # use pretrained embedding
    import tensorflow_hub as hub
    model = keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                       dtype=tf.string, input_shape=[], output_shape=[50]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam",
    metrics=["accuracy"])
  ```

* Neural Machine Translation

  ```python

    import tensorflow_addons as tfa
    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

    embeddings = keras.layers.Embedding(vocab_size, embed_size)
    encoder_embeddings = embeddings(encoder_inputs)
    decoder_embeddings = embeddings(decoder_inputs)

    encoder = keras.layers.LSTM(512, return_state=True) # return_state=True so that it can return its final hidden state and pass it to the decoder
    encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
    encoder_state = [state_h, state_c]

    sampler = tfa.seq2seq.sampler.TrainingSampler()
    # The traning sampler is one of  several samplers available in TF Addons, their role is to tell the decoder at each step what it should pretend the previous output was. During inference, this should be the embedding of the token that was actually output. During training, this shuld be the mebedding of the previous target token. 
    # In practice, it is often a good idea to start training with the embedding of the target of the previous time step and gradually transition to using the embedding of the actual token that was output at the previous step. # The ScheduledEmbeddingTrainingSampler will randomly choose between the target or the actual output, with a probability that you can gradu‐ ally change during training.

    decoder_cell = keras.layers.LSTMCell(512)
    output_layer = keras.layers.Dense(vocab_size)

    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                output_layer=output_layer)
    final_outputs, final_state, final_sequence_lengths = decoder(decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths)
    Y_proba = tf.nn.softmax(final_outputs.rnn_output)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
                        outputs=[Y_proba])

    # Bidirectional RNN
    keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))

    # Beam search
    beam_width = 10
    decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
        cell=decoder_cell, beam_width=beam_width, output_layer=output_layer)
    decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
        encoder_state, multiplier=beam_width)
    outputs, _, _ = decoder(
        embedding_decoder, start_tokens=start_tokens, end_token=end_token,
        initial_state=decoder_initial_state)
  ```
* Attention

  ```python
  # Luong attention usage
  attention_mechanism = tfa.seq2seq.attention_wrapper.LuongAttention(
         units, encoder_state, memory_sequence_length=encoder_sequence_length)
     attention_decoder_cell = tfa.seq2seq.attention_wrapper.AttentionWrapper(
         decoder_cell, attention_mechanism, attention_layer_size=n_units)

  """
  Transformer
  """
  class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
    super().__init__(dtype=dtype, **kwargs)
    if max_dims % 2 == 1: max_dims += 1 # max_dims must be even
    p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2)) pos_emb = np.empty((1, max_steps, max_dims))
    pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
    shape = tf.shape(inputs)
    return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]


  embed_size = 512; max_steps = 500; vocab_size = 10000
  encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
  decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
  embeddings = keras.layers.Embedding(vocab_size, embed_size)
  encoder_embeddings = embeddings(encoder_inputs)
  decoder_embeddings = embeddings(decoder_inputs)
  positional_encoding = PositionalEncoding(max_steps, max_dims=embed_size)
  encoder_in = positional_encoding(encoder_embeddings)
  decoder_in = positional_encoding(decoder_embeddings)



  Z = encoder_in for N in range(6):
        Z = keras.layers.Attention(use_scale=True)([Z, Z])
  encoder_outputs = Z Z = decoder_in
  for N in range(6):
        Z = keras.layers.Attention(use_scale=True, causal=True)([Z, Z])
        Z = keras.layers.Attention(use_scale=True)([Z, encoder_outputs])
        # use_scale=True: creates an additional parameter that lets the layer learn how to properly downscale the similarity score
        # causal=True ensure each output token only attends to the previous output tokens, not future ones. 
  outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(vocab_size, activation="softmax"))(Z) 
  ```
  