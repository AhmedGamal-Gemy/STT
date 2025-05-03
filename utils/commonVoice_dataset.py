import tensorflow as tf
from datasets import load_dataset, Dataset
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from .audio_processing import AudioProcessor
from .tokenizer import Tokenizer
import os

# Load the common voice dataset using hugging face using stream to not download the entire dataset.
def load_common_voice_stream(
        split: str = "train",
        language: str = "en",
        max_samples : int = None,
        cache_dir: str = "./data_cache"
) -> tf.data.Dataset :
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}_{language}")

    # Load from cache if exists
    if os.path.exists(cache_path):
        return Dataset.load_from_disk(cache_path)

    
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        language,
        split = split,
        streaming = True
    )

    # if max samples specfied take only that number of samples
    if max_samples:
        dataset = dataset.take(max_samples)

    full_dataset = Dataset.from_generator(lambda: dataset)
    full_dataset.save_to_disk(cache_path)
    return full_dataset

# preprocess one instance of data ( audio processing and tokenization )    
def preprocess_instance(
        instance : Dict,
        audio_processor : AudioProcessor,
        tokenizer : Tokenizer
) -> Tuple[tf.Tensor, tf.Tensor] :
    
    # Getting the audio array from the common voice data
    audio_array = np.array(instance['audio']['array'], dtype=np.float32 )

    # Process the array
    padded_audio = audio_processor.pad_audio(audio_array)

    processed_audio = audio_processor.audio_to_mfcc(padded_audio)

    if tf.math.reduce_any(tf.math.is_nan(processed_audio)):
        print("NaN detected in MFCC features")
        return None  # Will be filtered later

    # Process the text
    text = instance["sentence"].lower()
    processed_text = tokenizer.encode(text)

    # returning MFCC and Token IDs
    return processed_audio, tf.cast(processed_text, tf.int32)

# Create tensorflow dataset from the hugging face dataset and preprocess each instance return tf dataset
def create_tf_dataset(hf_dataset, audio_processor, tokenizer, batch_size=32):
    
    def generator():
        for instance in hf_dataset:
            try:
                audio = np.array(instance['audio']['array'], dtype=np.float32)
                text = instance['sentence'].lower()
                
                # Process audio
                padded_audio = audio_processor.pad_audio(audio)
                mfcc = audio_processor.audio_to_mfcc(padded_audio)
                
                # Process text
                tokens = tokenizer.encode(text)
                
                yield (mfcc, tokens)
            except Exception as e:
                print(f"Skipping invalid sample: {e}")
                continue

    # Create base dataset
    dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, audio_processor.n_mfcc)),  # MFCC
                tf.TensorSpec(shape=(None,), dtype=tf.int32)          # Labels
                )          
            )


    # Filtering and batching
    dataset = dataset.filter(lambda mfcc, labels: (
            tf.reduce_all(tf.math.is_finite(mfcc)) & 
            (tf.shape(labels)[0] > 0)
        )
    )
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [None, audio_processor.n_mfcc],  # MFCC
            [None]                           # Labels
        )
    )
    
    # Add dummy outputs for CTC loss later
    dataset = dataset.map(
        lambda mfcc, labels: ((mfcc, labels), tf.zeros(tf.shape(mfcc)[0]))
    )
    return (
        dataset
        .repeat()  # Keep providing data indefinitely
        .prefetch(tf.data.AUTOTUNE)
    )


def visualize_sample(
    mfcc: tf.Tensor,
    token_ids: tf.Tensor,
    tokenizer: Tokenizer,
    sample_rate: int = 16000
) -> None:
    """Plot MFCC features and decoded text for a single sample."""
    # Convert tensors to numpy
    mfcc_np = mfcc.numpy().T  # (n_mfcc, time)
    text = tokenizer.decode(token_ids)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot MFCCs
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        mfcc_np,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC Features")
    
    # Plot text
    plt.subplot(2, 1, 2)
    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.axis("off")
    plt.title("Decoded Text")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the pipeline
    audio_processor = AudioProcessor()
    tokenizer = Tokenizer()
    
    # Load a small subset
    hf_dataset = load_common_voice_stream(split="train", max_samples=10)
    
    # Build vocab from first 100 examples
    tokenizer.build_vocab(hf_dataset.take(100))

    print("Vocabulary size:", len(tokenizer.vocab))
    print("First few vocabulary items:", tokenizer.vocab[:10]) # Print the first 10 items

    # Create TF dataset
    tf_dataset = create_tf_dataset(hf_dataset, audio_processor, tokenizer, batch_size=2)
    
    # Visualize first batch
    for mfcc_batch, token_batch in tf_dataset.take(2):
        for i in range(mfcc_batch.shape[0]):
            visualize_sample(mfcc_batch[i], token_batch[i], tokenizer)
            
        print("MFCC batch shape:", mfcc_batch.shape)  # Should be (batch, time, n_mfcc)
        print("Token batch shape:", token_batch.shape) # (batch, max_token_length)