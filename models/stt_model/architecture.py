import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="models.stt_model")

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

    # REGULARIZATION
    l2_reg = tf.keras.regularizers.L2(5e-4)


    x = layers.Lambda(lambda x: (x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)) / 
                 (tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-6))(x)


    # CNN for pattern extraction, Padding is to add same padding up down right and left.
    # and this will result for the output shape to be the same as input shape

    # CNN Block with strided convolutions
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer = l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D( (2,1) ) (x) # Downsample that reduce spatial dimensions of input 

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer = l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D( (2,1) ) (x) # Downsample that reduce spatial dimensions of input 

    # Drop entire feature maps
    x = layers.SpatialDropout2D(0.4)(x)
 
    # Now reshaping for the RNN ( batch, time, feature ) by just feature * channels
    x = layers.Reshape( (-1, x.shape[2] * x.shape[3] ) ) (x)

    print("Reshaped features dimension:", x.shape)

    # BiLSTM Layers with regularization to avoid overfitting. Bidirection to capture information from both sides and 
    # LSTM because audio is sequential
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, 
                                      dropout=0.5, recurrent_dropout=0.4, kernel_regularizer = l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                      dropout=0.5, recurrent_dropout=0.4, kernel_regularizer = l2_reg))(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)  # Character-level modeling
    x = layers.BatchNormalization()(x)  

    # Time-distributed dense : adds non-linear feature transformation before CTC
    x = layers.TimeDistributed(layers.Dense(128, activation="relu", kernel_regularizer = l2_reg))(x)

    x = layers.Dropout(0.4)(x)

#   Custom bias initializer to balance character classes
    char_class_bias = np.zeros(vocab_size + 1)

    for i in range(3, vocab_size + 1):  # Skip pad, unk, blank
        char_class_bias[i] = -1.0

    # 1. Penalize blank token (prevents blank-dominated predictions)
    char_class_bias[2] = -10.0  # Blank token

    # 2. Penalize space token (prevents space-dominated predictions)
    char_class_bias[3] = -2.0  # Space token

    # 3. Penalize vowels (model is biased toward vowels)
    vowel_indices = [11, 15, 19, 25, 31]  # a, e, i, o, u
    for idx in vowel_indices:
        char_class_bias[idx] = -2.0

    # 4. Boost consonants (especially common ones)
    consonant_indices = [12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 
                        26, 27, 28, 29, 30, 32, 33, 34, 35, 36]  # All consonants
    common_consonants = [18, 30, 28, 29, 24]  # h, t, r, s, n
                        
    for idx in consonant_indices:
        char_class_bias[idx] = 1.0  # Slight boost for all consonants
        
    for idx in common_consonants:
        char_class_bias[idx] = 1.0  # Extra boost for common consonants

    # Use the custom bias initializer
    outputs = layers.Dense(
        vocab_size + 1,
        activation="linear",
        name="logits",
        bias_initializer=tf.keras.initializers.Constant(value=char_class_bias)
    )(x)

    return Model(inputs=input_layer, outputs=outputs)




# Create the CTC layer for Speech task. (CTC loss is automatically assigning portions of audio to labels and penlizes any predication that is incorrect)
class CTCLossLayer(layers.Layer):
    """Custom CTC loss layer with simplified blank penalty."""
    def __init__(self, blank_index=None, name="ctc_loss"):
        super().__init__(name=name)
        self.blank_index = blank_index

    def call(self, inputs):
        y_pred, y_true = inputs  # y_pred shape: (batch, time, vocab+1)
        
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.fill([batch_size], tf.shape(y_pred)[1])  # [batch_size]
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32),  # 0 = padding
            axis=1
        )
        
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index
        )

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
    
    # CTC layer
    ctc_layer = CTCLossLayer(blank_index=2)
    outputs = ctc_layer([logits, label_input])  # Logits first, labels second
    
    # Compile model
    model = Model(inputs=[mfcc_input, label_input], outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0,
        epsilon=1e-7
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