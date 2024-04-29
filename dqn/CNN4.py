import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, params):
        super(DQN, self).__init__()

        self.load_file = params['load_file']
        self.save_file = params['save_file']

        # Total params: 114580 (447.58 KB)
        # Trainable params: 114580 (447.58 KB)
        # Non-trainable params: 0 (0.00 Byte)

        # Block 1
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.ln1 = tf.keras.layers.LayerNormalization()

        # Fully connected
        self.fc1 = tf.keras.layers.Dense(units=16)
        self.lnf = tf.keras.layers.LayerNormalization()
        self.fc2 = tf.keras.layers.Dense(units=4)

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

        x1 = self.conv1(inputs)
        x1 = self.ln1(x1, training=training)
        x1 = tf.nn.relu(x1)

        x = tf.keras.layers.Flatten()(x1)
        x = self.fc1(x)
        x = self.lnf(x, training=training)
        x = tf.nn.relu(x)

        x = self.fc2(x)

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
