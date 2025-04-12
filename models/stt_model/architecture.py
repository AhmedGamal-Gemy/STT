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

    x = layers.Conv2D(32, (3,3), activation = "relu", padding = "same") (x)
    x = layers.MaxPooling2D( (1,2) ) (x) # Downsample that reduce spatial dimensions of input 

    # Now reshaping for RNN ( batch, time, feature ) by just feature * channels
    x = layers.Reshape( (-1, x.shape[2] * x.shape[3] ) ) (x)

    # BiLSTM layers. Bidirection to capture information from both sides and LSTM because audio is sequential
    x = layers.Bidirectional( layers.LSTM( 128, return_sequences = True ) ) (x)
    x = layers.Bidirectional( layers.LSTM( 128, return_sequences = True ) ) (x)

    # Output layer (logits for each token at every time step) and + 1 to add the blank
    outputs = layers.Dense(vocab_size + 1, activation="linear", name="logits")(x)

    return Model(inputs = input_layer, outputs = outputs)

# Create the CTC layer for Speech task. (CTC loss is automatically assign portions of audio to labels and penlizes any predication that is incorrect)
class CTCLossLayer(layers.Layer):
    """Custom CTC loss layer with proper dimension handling"""
    def __init__(self,blank_index, name="ctc_loss"):
        super().__init__(name=name)
        self.blank_index = blank_index
        
    def call(self, inputs):
        # Unpack inputs - order matters!
        # y_pred should come first (logits), then y_true (labels)
        y_pred, y_true = inputs  # y_pred shape: (batch, time, vocab+1)
        
        # Calculate sequence lengths
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.fill([batch_size], tf.shape(y_pred)[1])  # [batch_size]
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32),  # 0 = padding
            axis=1
        )
      
        
        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index  # Typically last index
        )
        self.add_loss(tf.reduce_mean(loss))
        return y_pred
    

    def compute_output_shape(self, input_shapes):
        # input_shapes is a list/tuple: (y_true_shape, y_pred_shape)
        return input_shapes[1]  # Return shape of y_pred

    def get_config(self):
        return {"blank_index": self.blank_index}



# Combine the basic model with CTC loss        
def create_STT_with_CTC(input_dim=13, vocab_size=37) -> Model:
    # Input layers
    mfcc_input = layers.Input(shape=(None, input_dim), name="MFCC_input")
    label_input = layers.Input(shape=(None,), dtype=tf.int32, name="Label_input")
    
    # Base model
    base_model = build_stt_model(input_dim, vocab_size)
    logits = base_model(mfcc_input)  # shape: (batch, time, vocab+1)
    
    # CTC layer - note the input order!
    ctc_layer = CTCLossLayer(blank_index=vocab_size)
    outputs = ctc_layer([logits, label_input])  # Logits first, labels second
    
    # Compile model
    model = Model(inputs=[mfcc_input, label_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
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