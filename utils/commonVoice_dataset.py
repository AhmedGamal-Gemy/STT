import tensorflow as tf
from datasets import load_dataset
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from .audio_processing import AudioProcessor
from .tokenizer import Tokenizer


# Load the common voice dataset using hugging face using stream to not download the entire dataset.
def load_common_voice_stream(
        split: str = "train",
        language: str = "en",
        sample_rate : int = 16000,
        max_samples : int = None,
) -> tf.data.Dataset :
    
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        language,
        split = split,
        streaming = True
    )

    # if max samples specfied take only that number of samples
    if max_samples:
        dataset = dataset.take(max_samples)
    
    return dataset

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
def create_tf_dataset(
        hf_dataset,
        audio_processor : AudioProcessor,
        tokenizer : Tokenizer,
        batch_size : int = 32

) -> tf.data.Dataset:
    
    def generator():
        for instance in hf_dataset:
            # yield Used because it allow further execution unlike return
            yield preprocess_instance(instance,audio_processor, tokenizer)

    # Create the tensorflow dataset from the generator

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature = (       # Specify the shape and the type of the output
            tf.TensorSpec(shape = ( None, audio_processor.n_mfcc ), dtype = tf.float32 ),    # For the audio part
            tf.TensorSpec(shape = (None), dtype = tf.int32)     # For the text part
        )   
    )
    # Filter out invalid samples here
    dataset = dataset.filter(
        lambda mfcc, text: 
            tf.logical_and(
                tf.reduce_all(tf.math.is_finite(mfcc)),  # Check finite values
                tf.greater(tf.shape(text)[0], 0)          # Check non-empty text
            )
    )
     # Dynamic padding for variable-length sequences

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [None, audio_processor.n_mfcc],  # MFCC features (time, n_mfcc)
            [None]                           # Token IDs (time,)
        )
    )

    # Restructure to ((mfcc, labels), dummy_target) where dummy_target is zeros
    # Add dummy output and optimize
    dataset = dataset.map(
        lambda mfcc, labels: ((mfcc, labels), tf.zeros(tf.shape(labels)[0]))
    ).repeat().prefetch(tf.data.AUTOTUNE)

    return dataset



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



