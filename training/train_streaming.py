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
BATCH_SIZE = 64
MAX_TRAIN_SAMPLES = 1_000

# wer and cer metrics
class STTMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_batch, tokenizer):
        super().__init__()
        self.val_batch = val_batch
        self.tokenizer = tokenizer
        self.blank_index = 2 
        print("STTMetrics initialized with AGGRESSIVE token forcing")

    def on_epoch_end(self, epoch, logs=None):
        print("\n=== CTC DEBUG START ===")
        try:
            # Use the provided validation batch
            ((audio_inputs, true_labels), _) = self.val_batch
                
            # Get base model
            base_model = next(layer for layer in self.model.layers 
                            if isinstance(layer, tf.keras.Model))
            
            # Predict
            pred_logits = base_model.predict(audio_inputs, verbose=0)
            
            # Print top predictions for debugging
            log_probs = tf.nn.log_softmax(pred_logits, axis=-1)
            top_chars = tf.math.top_k(log_probs[0, 0, :], k=5)
            print(f"| Top 5 predictions at first timestep: {top_chars}")
            
            # PROPER CTC DECODING - no extreme penalties
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(pred_logits, [1, 0, 2]),
                sequence_length=[pred_logits.shape[1]] * pred_logits.shape[0],
                beam_width=100,
                top_paths=1
            )
            
            pred_ids = decoded[0]
            
            # Convert to text
            texts_pred = []
            for ids in tf.sparse.to_dense(pred_ids).numpy():
                # Filter out padding and blank tokens
                valid_ids = [idx for idx in ids if idx > 0 and idx != self.blank_index]
                if not valid_ids:  # If empty prediction
                    texts_pred.append("[empty]")
                else:
                    texts_pred.append(self.tokenizer.decode(valid_ids))
            
            texts_true = []
            for label in true_labels:
                # Filter out padding
                valid_ids = [idx for idx in label if idx > 0]
                texts_true.append(self.tokenizer.decode(valid_ids))
            
            # Calculate metrics
            logs["wer"] = wer(texts_true, texts_pred)
            logs["cer"] = cer(texts_true, texts_pred)
            
            print(f"| Pred shape: {pred_ids.shape}")
            for i in range(min(5, len(texts_pred))):
                print(f"| Example {i+1}:")
                print(f"|   True: {texts_true[i]}")
                print(f"|   Pred: {texts_pred[i]}")
            print(f"| WER: {logs['wer']:.2%} | CER: {logs['cer']:.2%}")

        except Exception as e:
            print(f"| ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        print("=== CTC DEBUG END ===\n")

class ClassBalanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, vowel_indices, blank_index=2, space_index=3):
        super().__init__()
        self.vowel_indices = vowel_indices
        self.blank_index = blank_index
        self.space_index = space_index
        
    def on_epoch_begin(self, epoch, logs=None):
        # Find the logits layer
        logits_layer = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):  # Base model
                for base_layer in layer.layers:
                    if isinstance(base_layer, tf.keras.layers.Dense) and base_layer.name == "logits":
                        logits_layer = base_layer
                        break
        
        if logits_layer:
            # Get current weights
            weights, biases = logits_layer.get_weights()
            
            # Apply balance adjustments at each epoch
            # The strength increases with epochs to help guide learning
            vowel_penalty = -0.7  
            blank_penalty = -0.2
            
            # Apply penalties
            biases[self.blank_index] = blank_penalty
            biases[self.space_index] = 0.5
            
            for idx in self.vowel_indices:
                biases[idx] = vowel_penalty
                
            # Set the adjusted weights back
            logits_layer.set_weights([weights, biases])
            print(f"Epoch {epoch+1}: Applied class balance adjustments")



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
    ).repeat()
    
    return train_dataset, test_dataset

def train_model():
    setup_directories()
    tokenizer = Tokenizer(max_text_length=50)
    train_dataset, test_dataset = get_datasets(tokenizer)

    # Create a materialized validation dataset
    print("Creating materialized validation set...")
    real_val_samples = []
    
    # Use a Python try/finally pattern to ensure dataset is properly closed
    try:
        # Take 5 batches from the validation set
        for i, batch in enumerate(test_dataset.take(5)):
            real_val_samples.append(batch)
            print(f"Collected validation batch {i+1}")
            if i >= 4:  # Stop after 5 batches
                break
    except Exception as e:
        print(f"Error collecting validation data: {e}")
    
    # Use the first batch for the metrics callback if available
    if real_val_samples:
        real_val_batch = real_val_samples[0]
        print(f"Using real validation data with shape: {real_val_batch[0][0].shape}")
    else:
        # Fallback to synthetic data only if necessary
        print("WARNING: Using synthetic validation data")
        import numpy as np
        
        # Create synthetic data matching your dataset format
        synth_audio = np.random.normal(size=(16, 376, 13)).astype(np.float32)
        synth_text_ids = np.zeros((16, 50), dtype=np.int32)
        example_text = "hello world"
        text_encoded = tokenizer.encode(example_text)
        for i, idx in enumerate(text_encoded[:50]):
            synth_text_ids[0, i] = idx
        synth_targets = np.ones((16,), dtype=np.int32)
        real_val_batch = ((synth_audio, synth_text_ids), synth_targets)

    # Initialize model
    model = create_STT_with_CTC(
        input_dim=13,
        vocab_size=len(tokenizer.vocab)
    )

    vowel_indices = [11, 15, 19, 25, 31]


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/model_{epoch:02d}.keras",
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        ),
        STTMetrics(real_val_batch, tokenizer),  # Use real validation data
        ClassBalanceCallback(vowel_indices)
    ]

    # Rest of your code remains the same
    steps_per_epoch = MAX_TRAIN_SAMPLES // BATCH_SIZE
    validation_steps = (MAX_TRAIN_SAMPLES // 2) // BATCH_SIZE

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


def train_with_curriculum(model, train_dataset, test_dataset, epochs=20, callbacks=None):
    """Train the model using curriculum learning strategy."""
    print("Starting curriculum learning training...")
    
    # Extract examples from dataset to sort them
    print("Extracting examples from dataset...")
    examples = []
    targets = []
    
    # Take all examples from the dataset
    for batch in train_dataset:
        # Structure is ((audio, labels), targets)
        (audios, labels), batch_targets = batch
        
        # Convert to numpy for consistent handling
        audios_np = audios.numpy()
        labels_np = labels.numpy()
        targets_np = batch_targets.numpy()
        
        # Store individual examples
        for i in range(len(audios_np)):
            examples.append((audios_np[i], labels_np[i]))
            targets.append(targets_np[i])
    
    print(f"Extracted {len(examples)} examples")
    
    # Sort examples by audio length (shorter first)
    # Get audio length by finding non-zero values in first feature dimension
    lengths = [np.sum(np.abs(ex[0][:, 0]) > 1e-6) for ex in examples]
    sorted_indices = np.argsort(lengths)
    
    # Reorder examples and targets
    sorted_examples = [examples[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]
    
    print("Examples sorted by length")
    
    # Define curriculum stages
    stages = [
        {"name": "Easy (short utterances)", "fraction": 0.25, "epochs": 5},
        {"name": "Medium", "fraction": 0.6, "epochs": 5},
        {"name": "Full dataset", "fraction": 1.0, "epochs": 10}
    ]
    
    # Initialize history to collect metrics
    history = {"loss": [], "val_loss": [], "wer": [], "cer": []}
    
    # Train in stages
    for stage in stages:
        print(f"\n===== CURRICULUM STAGE: {stage['name']} =====")
        
        # Calculate how many examples to use in this stage
        stage_size = int(len(sorted_examples) * stage["fraction"])
        print(f"Using {stage_size} examples ({stage['fraction']*100:.0f}% of dataset)")
        
        # Get examples for this stage
        stage_examples = sorted_examples[:stage_size]
        stage_targets = sorted_targets[:stage_size]
        
        # Create a TensorFlow dataset from these examples
        # Convert back to the expected format: ((audio, labels), targets)
        stage_audios = np.stack([ex[0] for ex in stage_examples])
        stage_labels = np.stack([ex[1] for ex in stage_examples])
        stage_targets = np.array(stage_targets)
        
        # Create dataset
        stage_dataset = tf.data.Dataset.from_tensor_slices(
            ((stage_audios, stage_labels), stage_targets)
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Train for this stage
        print(f"Training for {stage['epochs']} epochs...")
        stage_history = model.fit(
            stage_dataset, 
            epochs=stage["epochs"],
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Collect metrics
        for key in stage_history.history:
            if key in history:
                history[key].extend(stage_history.history[key])
        
        print(f"Completed {stage['name']} stage")
    
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