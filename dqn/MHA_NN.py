import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, params):
        super(DQN, self).__init__()

        self.load_file = params['load_file']
        self.save_file = params['save_file']

        # Total params: 230068 (898.70 KB)
        # Trainable params: 230068 (898.70 KB)
        # Non-trainable params: 0 (0.00 Byte)

        self.embedding_layer = ChannelEmbeddingLayer(6, 16)

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)

        self.fc1 = tf.keras.layers.Dense(units=16)
        self.lnf = tf.keras.layers.LayerNormalization()

        # # Fully connected
        self.fcEnd = tf.keras.layers.Dense(units=16, activation='relu')
        self.out = tf.keras.layers.Dense(units=4, activation='linear')

        self.initial_learning_rate = params['lr']
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=80000,
            decay_rate=0.25,
            staircase=True)

        # Optimizer
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.lr_schedule)

        # Checkpointing
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.save_file, max_to_keep=5)

        # Training step counter
        self.train_step_counter = tf.Variable(0, trainable=False, name='train_step_counter')

    def call(self, x, training):

        x = self.embedding_layer(x)

        x = self.mha(x, x)

        x = tf.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = self.lnf(x, training=training)
        x = tf.nn.relu(x)
        x= self.fcEnd(x)

        x = self.out(x)
        return x

    @tf.function
    def train_step(self, bat_s, bat_a_onehot, yj, is_weights):
        bat_a_onehot = tf.cast(bat_a_onehot, dtype=tf.float32)
        yj = tf.cast(yj, dtype=tf.float32)
        is_weights      = tf.cast(is_weights, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self(bat_s, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, bat_a_onehot), axis=1)
            td_errors = yj - q_action
            loss = tf.reduce_mean(is_weights * tf.square(td_errors))

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Clip gradients
        clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

        # Apply clipped gradients
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))

        # Increment training step counter
        self.train_step_counter.assign_add(1)

        return loss, td_errors


    def load_checkpoint(self, specific_path=None):
        specific_path = specific_path or self.load_file
        if specific_path:
            self.checkpoint.restore(specific_path).expect_partial()
            print(f"Checkpoint restored from {specific_path}.")
        elif self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print("Latest checkpoint restored.")
        else:
            print("No checkpoint found. Initializing model from scratch.")

        self.train_step_counter.assign(0)  # Reset the training step counter
        print(f"Counter reinitialized: {self.train_step_counter.numpy()}")

class ChannelEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, num_channels, embedding_dim):
        super(ChannelEmbeddingLayer, self).__init__()
        # Initialize the embeddings for each channel
        self.channel_embeddings = self.add_weight(
            shape=(num_channels, embedding_dim),
            initializer='uniform',
            trainable=True
        )

    def call(self, inputs):
        channel_indices = tf.range(inputs.shape[-1])
        embeddings = tf.nn.embedding_lookup(self.channel_embeddings, channel_indices)

        embeddings_expanded = tf.reshape(embeddings, (1, 1, 1, inputs.shape[-1], -1))

        inputs_expanded = tf.expand_dims(inputs, -1)

        embedded_inputs = inputs_expanded * embeddings_expanded

        output = tf.reduce_sum(embedded_inputs, axis=-2)

        return output
