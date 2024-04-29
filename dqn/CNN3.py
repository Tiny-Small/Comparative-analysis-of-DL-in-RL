import numpy as np
import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, params):
        super(DQN, self).__init__()

        self.load_file = params['load_file']
        self.save_file = params['save_file']

        # Total params: 85972 (335.83 KB)
        # Trainable params: 85972 (335.83 KB)
        # Non-trainable params: 0 (0.00 Byte)

        # Block 1
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.ln1 = tf.keras.layers.LayerNormalization()

        # Projection for Block 1 to Block 2 skip connection
        self.conv_proj1_to_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.ln_proj1_to_2 = tf.keras.layers.LayerNormalization()  # Improve stability
        self.relu_proj1_to_2 = tf.keras.layers.ReLU()  # Adding non-linearity

        # Block 2
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.ln2 = tf.keras.layers.LayerNormalization()

        # Projection for Block 2 to Block 3 skip connection
        self.conv_proj2_to_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), padding='same')
        self.ln_proj2_to_3 = tf.keras.layers.LayerNormalization()  # Improve stability
        self.relu_proj2_to_3 = tf.keras.layers.ReLU()  # Adding non-linearity

        # Block 3
        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')
        self.ln3 = tf.keras.layers.LayerNormalization()

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

        # Block 1
        x1 = self.conv1(inputs)
        x1 = self.ln1(x1, training=training)
        x1 = tf.nn.relu(x1)

        # skip connection 1
        x_shortcut = self.conv_proj1_to_2(x1)
        x_shortcut = self.ln_proj1_to_2(x_shortcut, training=training)
        x_shortcut = self.relu_proj1_to_2(x_shortcut)

        # block 2
        x1 = self.conv2(x1)
        x1 += x_shortcut # skip connection 1 added to block 2
        x1 = self.ln2(x1, training=training)
        x1 = tf.nn.relu(x1)

        # skip connection 2
        x_shortcut = self.conv_proj2_to_3(x1)
        x_shortcut = self.ln_proj2_to_3(x_shortcut, training=training)
        x_shortcut = self.relu_proj2_to_3(x_shortcut)

        # block 3
        x1 = self.conv3(x1)
        x1 += x_shortcut # skip connection 2 added to block 3
        x1 = self.ln3(x1, training=training)
        x1 = tf.nn.relu(x1)

        # Block 7
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
