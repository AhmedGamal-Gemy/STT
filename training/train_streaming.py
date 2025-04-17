import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.commonVoice_dataset import load_common_voice_stream, create_tf_dataset
from utils.audio_processing import AudioProcessor
from utils.tokenizer import Tokenizer
from models.stt_model.architecture import create_STT_with_CTC
from jiwer import wer, cer

# Configuration
tf.keras.mixed_precision.set_global_policy('float32')

SAMPLE_RATE = 16000
EPOCHS = 10
BATCH_SIZE = 8
MAX_TRAIN_SAMPLES = 2000  # Start small for testing

# wer and cer metrics
class STTMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, tokenizer, samples=5):
        super().__init__()
        self.val_data = val_dataset
        self.tokenizer = tokenizer
        self.samples = samples

    def on_epoch_end(self, epoch, logs=None):
        print("\n=== CTC DEBUG START ===")
        try:
            # Get validation batch
            val_batch = next(iter(self.val_data.take(1)))
            ((audio_inputs, true_labels), _) = val_batch
            
            # Get base model
            base_model = next(layer for layer in self.model.layers 
                            if isinstance(layer, tf.keras.Model))
            
            # Predict
            pred_logits = base_model.predict(audio_inputs, verbose=0)
            
            # CTC decode
            decoded, _ = tf.keras.backend.ctc_decode(
                pred_logits,
                input_length=[pred_logits.shape[1]] * pred_logits.shape[0],
                greedy=True
            )
            pred_ids = decoded[0]

            # Convert to numpy arrays properly
            texts_pred = [
                self.tokenizer.decode(tf.constant(ids))  # Convert numpy array to tensor
                for ids in pred_ids.numpy()
            ]
            texts_true = [
                self.tokenizer.decode(label) 
                for label in true_labels
            ]
            
            # Calculate metrics
            logs["wer"] = wer(texts_true, texts_pred)
            logs["cer"] = cer(texts_true, texts_pred)
            
            print(f"| Pred shape: {pred_ids.shape}")
            print(f"| Sample pred: {texts_pred[0]}")
            print(f"| Sample true: {texts_true[0]}")
            print(f"| WER: {logs['wer']:.2%} | CER: {logs['cer']:.2%}")

        except Exception as e:
            print(f"| ERROR: {str(e)}")
            raise
            
        print("=== CTC DEBUG END ===\n")


def setup_directories():
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def get_datasets(tokenizer: Tokenizer):
    """Load and preprocess datasets with proper CTC formatting"""
    train_hf = load_common_voice_stream("train", max_samples=MAX_TRAIN_SAMPLES)
    test_hf = load_common_voice_stream("validation", max_samples=MAX_TRAIN_SAMPLES//2)

    print("Building tokenizer vocabulary...")
    tokenizer.build_vocab(train_hf.take(32))  # Build from first 100 samples

    audio_processor = AudioProcessor()  
    
    # Create datasets with proper CTC formatting
    train_dataset = create_tf_dataset(
        train_hf, 
        audio_processor,
        tokenizer,
        batch_size=BATCH_SIZE
    )
    test_dataset = create_tf_dataset(
        test_hf,
        audio_processor,
        tokenizer,
        batch_size=BATCH_SIZE
    )
    
    return train_dataset, test_dataset

def train_model():
    setup_directories()
    tokenizer = Tokenizer(max_text_length=50)  # Limit text length
    train_dataset, test_dataset = get_datasets(tokenizer)

    # Initialize model with correct blank index
    model = create_STT_with_CTC(
        input_dim=13,
        vocab_size=len(tokenizer.vocab)  # Blank will be at len(vocab)
    )

    # Verify blank index matches vocabulary
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Blank token index: {len(tokenizer.vocab)}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/model_{epoch:02d}.keras",
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            update_freq='epoch'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # More aggressive learning rate reduction
        patience=2,
        min_lr=1e-6,
        verbose=1
        ),
        STTMetrics(test_dataset, tokenizer)
    ]

    # Calculate steps per epoch
    steps_per_epoch = MAX_TRAIN_SAMPLES // BATCH_SIZE
    validation_steps = (MAX_TRAIN_SAMPLES // 2) // BATCH_SIZE

    print("Final logits shape:", model.output_shape)

    print("Starting training...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.savefig("logs/training_curve.png")
    plt.show()

if __name__ == "__main__":
    # Verify GPU availability
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Start training
    history = train_model()
    plot_training(history)
    
    # Save final model
    
    print("Training complete. Models saved to checkpoints")