import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, params):
        super(DQN, self).__init__()

        self.load_file = params['load_file']
        self.save_file = params['save_file']

        # Total params: 202472 (790.91 KB)
        # Trainable params: 202472 (790.91 KB)
        # Non-trainable params: 0 (0.00 Byte)

        self.conv2plus1d = Conv2Plus1D(spatial_filters=64, temporal_filters=64, final_channels=64)

        self.fc1 = tf.keras.layers.Dense(units=16)
        self.lnf = tf.keras.layers.LayerNormalization()

        # # Fully connected
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

    def call(self, inputs, training=False):

        x = self.conv2plus1d(inputs)

        x = tf.squeeze(x, axis=1)

        x = tf.keras.layers.Flatten()(x)
        x = self.fc1(x)
        x = self.lnf(x, training=training)
        x = tf.nn.relu(x)

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
    def __init__(self, num_channels, embedding_dim, height, width, name=None):
        super(ChannelEmbeddingLayer, self).__init__(name=name)
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.channel_embeddings = self.add_weight(
            name="channel_embeddings",
            shape=(num_channels, embedding_dim),
            initializer='uniform',
            trainable=True
        )
        self.positional_embeddings = self.add_weight(
            name="positional_embeddings",
            shape=(1, height, width, embedding_dim),
            initializer='uniform',
            trainable=True
        )

    def call(self, inputs):

        channel_indices = tf.range(inputs.shape[-1])
        channel_embeddings = tf.nn.embedding_lookup(self.channel_embeddings, channel_indices)

        channel_embeddings = tf.reshape(channel_embeddings, (1, 1, 1, 1, self.num_channels, self.embedding_dim))
        inputs_expanded = tf.expand_dims(inputs, -1)

        embedded_inputs = inputs_expanded * channel_embeddings
        output = tf.reduce_sum(embedded_inputs, axis=-2)

        output += self.positional_embeddings

        output = tf.transpose(output, [0, 2, 3, 1, 4])

        return output


class Conv2Plus1D(tf.keras.layers.Layer):
    def __init__(self, spatial_filters, temporal_filters, final_channels):
        super(Conv2Plus1D, self).__init__()
        # First part: Spatial Convolution
        self.spatial_conv = tf.keras.layers.Conv3D(
            filters=spatial_filters,
            kernel_size=(1, 3, 3),
            padding='same',
            activation='relu'
        )

        # Second part: Temporal Convolution
        self.temporal_conv = tf.keras.layers.Conv3D(
            filters=temporal_filters,
            kernel_size=(64, 1, 1),
            strides=(64, 1, 1),
            padding='valid',
            activation='relu'
        )

        # Reduce channels to the desired number
        self.channel_reducer = tf.keras.layers.Conv3D(
            filters=final_channels,
            kernel_size=(1, 1, 1),
            padding='same',
            activation='relu'
        )

    def call(self, inputs):
        x = self.spatial_conv(inputs)
        x = self.temporal_conv(x)
        x = self.channel_reducer(x)
        return x
