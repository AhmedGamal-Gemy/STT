import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from utils.commonVoice_dataset import load_common_voice_stream, create_tf_dataset, visualize_sample
from utils.audio_processing import AudioProcessor
from utils.tokenizer import Tokenizer
from models.stt_model.architecture import create_STT_with_CTC

# Configuration
tf.keras.mixed_precision.set_global_policy('float32')

SAMPLE_RATE = 16000
EPOCHS = 20
BATCH_SIZE = 32
MAX_TRAIN_SAMPLES = 1000

# Creating folders for logs and checkpoints
def setup_directories():
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

# Load and preprocess train/test data
def get_datasets(tokenizer: Tokenizer):
    
    # Load small datasets for testing
    train_hf = load_common_voice_stream("train",max_samples=MAX_TRAIN_SAMPLES)
    test_hf = load_common_voice_stream("validation",max_samples=MAX_TRAIN_SAMPLES)

    print("Building tokenizer vocablary.....")
    tokenizer.build_vocab( train_hf.take(100) )

    # Create tensorflow datasets
    audio_processor = AudioProcessor()
    train_dataset = create_tf_dataset(train_hf,audio_processor,tokenizer,batch_size=BATCH_SIZE).repeat()
    test_dataset = create_tf_dataset(test_hf,audio_processor,tokenizer,batch_size=BATCH_SIZE)

    return train_dataset, test_dataset


def train_model():
    # Initlize components
    setup_directories()
    tokenizer = Tokenizer()
    train_dataset, test_dataset = get_datasets(tokenizer)

    # Creating the model
    model = create_STT_with_CTC(
        input_dim=13,
        vocab_size= len(tokenizer.vocab) + 1 
    )

        # --- VERIFICATION ---
    print("\n=== Pre-Training Verification ===")
    print("Vocab size:", len(tokenizer.vocab))
    print("Blank token index:", len(tokenizer.vocab))  # Should match CTC config
    
    # Verify first training batch
    for batch in train_dataset.take(1):
        mfcc_batch, label_batch = batch[0]  # Unpack (inputs, dummy_outputs)
        print("\nMFCC batch shape:", mfcc_batch.shape)  # Should be (batch, time, 13)
        print("Labels batch shape:", label_batch.shape) # Should be (batch, text_len)
        print("Sample labels:", label_batch[0].numpy()) # Show first sample's tokens
    
    input("Press Enter to start training...")  # Pause for inspection
    # ----------------------------------------

    # Specify check points and tensorboard logs
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/model_{epoch:02d}.keras",
            save_best_only = True
        ),
        tf.keras.callbacks.TensorBoard(log_dir= "logs")
    ]

    # Adding learning rate scheduling
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1))

    # Training the model
    print("Starting training....")
    history = model.fit(
        train_dataset.repeat(),  # Add .repeat() here
        steps_per_epoch=MAX_TRAIN_SAMPLES//BATCH_SIZE,  # Match data size
        validation_steps=50,
        validation_data = test_dataset,
        epochs = EPOCHS,
        callbacks = callbacks
    )
    
    return history


def plot_training(history):
    """Plot training/validation loss."""
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("logs/training_plot.png")
    plt.show()

if __name__ == "__main__":
    # Phase 1: Architecture Validation
    print("\n=== Testing Model Architecture ===")
    dummy_mfcc = tf.random.normal((2, 100, 13))  # Test batch
    dummy_labels = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    
    test_model = create_STT_with_CTC(vocab_size=37)  # Match your test vocab size
    test_output = test_model.predict([dummy_mfcc, dummy_labels])
    print("Architecture test passed! Output shape:", test_output.shape)
    
    # Phase 2: Actual Training (uncomment when ready)
    print("\n=== Starting Real Training ===")
    history = train_model()
    plot_training(history)
    print("Training complete! Check checkpoints/ and logs/")