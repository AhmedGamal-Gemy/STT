import tensorflow as tf
from tensorflow.keras import layers, Model

# Build the basic model architecture. I will use CNN to capture patterns and BiLSTM to learn
# It gets input dimension ( which in out case number of MFCC ) and vocab size from tokenizer ( add 1 for blank )
# And return Model from keras
def build_stt_model(
        input_dim : int = 13,
        vocab_size : int = 37
        ) -> Model:
    
# Important note Keras implicitly add another dimension at the beginning which is batch size


    # Here the input layer takes shape as timesteps ( the features itself what happed in each second )
    # that's unknown and putted None to do this and input dim (in other words number of mfcc)

    # ( batch size, timesteps, n_mfcc )
    input_layer = layers.Input( shape = (None, input_dim), name = "Input_mfcc" )

    # To use CNN to capture pattern i should convert to shape suitable for CNN by
    # by add another dimension for the channels using reshape

    # (batch size, timesteps, feature, channels)
    x = layers.Reshape( ( -1, input_dim, 1) ) (input_layer)

    # CNN for pattern extraction, Padding is to add same padding up down right and left.
    # and this will result for the output shape to be the same as input shape

    # CNN Block with strided convolutions
    x = layers.Conv2D(32, (3,3), strides=(1,2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D( (1,2) ) (x) # Downsample that reduce spatial dimensions of input 

    x = layers.Conv2D(64, (3,3), strides=(1,2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D( (1,2) ) (x) # Downsample that reduce spatial dimensions of input 
 
    # Now reshaping for RNN ( batch, time, feature ) by just feature * channels
    x = layers.Reshape( (-1, x.shape[2] * x.shape[3] ) ) (x)

    print("Reshaped features dimension:", x.shape)

    # BiLSTM Layers with regularization to avoid overfitting. Bidirection to capture information from both sides and LSTM because audio is sequential
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, 
                                      dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True,
                                      dropout=0.3, recurrent_dropout=0.2))(x)

    # Time-distributed dense : adds non-linear feature transformation before CTC
    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.Dropout(0.4)(x)

    # # Initialize embeddings with some prior knowledge
    # char_embed_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    # outputs = layers.Dense(
    #     vocab_size + 1,
    #     activation="linear",
    #     bias_initializer="zeros",
    #     kernel_initializer=char_embed_init,
    #     name="logits"
    # )(x)

    # Output layer with bias against blank token
    blank_bias_initializer = tf.keras.initializers.Constant(
        value=[-1.0 if i == vocab_size else 0.0 for i in range(vocab_size + 1)]
    )
    
    outputs = layers.Dense(
        vocab_size + 1, 
        activation="linear", 
        name="logits",
        bias_initializer=blank_bias_initializer
    )(x)

    return Model(inputs=input_layer, outputs=outputs)




# Create the CTC layer for Speech task. (CTC loss is automatically assign portions of audio to labels and penlizes any predication that is incorrect)
class CTCLossLayer(layers.Layer):
    """Custom CTC loss layer with blank penalty."""
    def __init__(self, blank_index=None, name="ctc_loss"):
        super().__init__(name=name)
        self.blank_index = blank_index

    def call(self, inputs):
        y_pred, y_true = inputs  # y_pred shape: (batch, time, vocab+1)

        # DEBUGGING
        tf.debugging.assert_rank(y_pred, 3, message="Logits must be 3D (batch, time, vocab)")
        tf.debugging.assert_rank(y_true, 2, message="Labels must be 2D (batch, labels)")
        
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.fill([batch_size], tf.shape(y_pred)[1])  # [batch_size]
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32),  # 0 = padding
            axis=1
        )
        
        # Apply a fixed penalty to blank token
        blank_penalty_value = 25.0  # This forces the model away from blank predictions
        blank_mask = tf.one_hot(indices=[self.blank_index], depth=tf.shape(y_pred)[-1])
        blank_mask = tf.reshape(blank_mask, [1, 1, -1])  # Shape: [1, 1, vocab_size+1]
        
        # Subtract the penalty from blank logits
        y_pred_adjusted = y_pred - (blank_mask * blank_penalty_value)
        
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred_adjusted,  # Use the adjusted logits with blank penalty
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index
        )

        # Add small epsilon to avoid log(0)
        self.add_loss(tf.reduce_mean(loss) + 1e-7)
        return y_pred  # Return original predictions for the next layer

    def get_config(self):
        return {"blank_index": self.blank_index}

# Combine the basic model with CTC loss        
def create_STT_with_CTC(input_dim=13, vocab_size=37) -> Model:

    # tf.config.optimizer.set_jit(True)

    # Input layers
    mfcc_input = layers.Input(shape=(None, input_dim), name="MFCC_input")
    label_input = layers.Input(shape=(None,), dtype=tf.int32, name="Label_input")
    
    # Base model
    base_model = build_stt_model(input_dim, vocab_size)
    logits = base_model(mfcc_input)  # shape: (batch, time, vocab+1)
    
    # CTC layer - note the input order!
    ctc_layer = CTCLossLayer(blank_index=2)
    outputs = ctc_layer([logits, label_input])  # Logits first, labels second
    
    # Compile model
    model = Model(inputs=[mfcc_input, label_input], outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4,
        clipnorm=1.0
    )

    model.compile(optimizer=optimizer)
    
    return model

    
# Visualize the architecture
def visualize_model(model: Model):
    """Plot model architecture."""
    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        to_file="stt_model.png"
    )

if __name__ == "__main__":
    # Test the model
    model = create_STT_with_CTC(input_dim=13, vocab_size=37)
    model.summary()
    # visualize_model(model)
    
    # Test with dummy data
    dummy_mfcc = tf.random.normal((2, 100, 13))  # Batch size 2, timesteps 100, features 13
    dummy_labels = tf.cast(tf.constant([[1, 2, 3], [4, 5, 6]]), tf.float32)  # Cast to float32
    
    # Forward pass
    outputs = model.predict([dummy_mfcc, dummy_labels])
    print(outputs)
    print("Test successful!")