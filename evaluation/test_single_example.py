import tensorflow as tf
import numpy as np
from utils.audio_processing import AudioProcessor
from utils.tokenizer import Tokenizer
from models.stt_model.architecture import CTCLossLayer
from utils.commonVoice_dataset import load_common_voice_stream, create_tf_dataset

def decode_greedy(logits, blank_index=2):
    """Simple greedy decoder that collapses repeated tokens and removes blanks."""
    # Get the most likely token at each timestep
    predictions = np.argmax(logits, axis=-1)
    
    # Collapse repeated characters and remove blanks
    decoded = []
    prev = -1  # Initialize with an impossible token value
    
    for p in predictions:
        if p != prev and p != blank_index:  # Not a repeat and not blank
            decoded.append(p)
        prev = p
        
    return decoded

def test_single_example():
    # Set up tokenizer
    tokenizer = Tokenizer(max_text_length=50)
    train_hf = load_common_voice_stream("train", max_samples=100)
    
    def prepare_for_tokenizer(batch):
        text = batch['sentence'] if 'sentence' in batch else batch['text']
        return {"sentence": text}
    
    tokenizer_dataset = train_hf.map(prepare_for_tokenizer)
    print("Building tokenizer vocabulary...")
    tokenizer.build_vocab(tokenizer_dataset)
    
    # Print vocabulary
    print("\nTokenizer vocabulary:")
    for i, token in enumerate(tokenizer.vocab):
        print(f"Index {i}: '{token}'")
    
    # Load model
    model_path = "checkpoints/model_01.keras"
    model = tf.keras.models.load_model(
        model_path, 
        compile=False,
        custom_objects={"CTCLossLayer": CTCLossLayer}
    )
    
    # Get the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        base_model = model
    
    print(f"Found base model: {base_model.name}")
    
    # Get an audio sample
    audio_processor = AudioProcessor()
    train_dataset = create_tf_dataset(
        train_hf, 
        audio_processor,
        tokenizer,
        batch_size=32
    )
    
    # Take a single example
    for batch in train_dataset.take(1):
        (audio_features, text_labels), _ = batch
        single_audio = audio_features[0:1]
        true_label = text_labels[0]
        break
    
    # Run prediction
    print(f"Input audio shape: {single_audio.shape}")
    logits = base_model.predict(single_audio)
    print(f"Output logits shape: {logits.shape}")
    
    # Display top token predictions at sample timesteps
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    timestep_interval = logits.shape[1] // 10
    
    print("\nTop token predictions at sample timesteps:")
    print("Time | Top 1 | Top 2 | Top 3")
    print("-" * 30)
    
    for t in range(0, logits.shape[1], timestep_interval):
        if t >= logits.shape[1]:
            continue
        
        top_k = tf.math.top_k(log_probs[0, t], k=3)
        top_indices = top_k.indices.numpy()
        top_values = top_k.values.numpy()
        
        tokens = []
        for idx in top_indices:
            if idx == 2:
                tokens.append("BLANK")
            elif 0 <= idx < len(tokenizer.vocab):
                tokens.append(tokenizer.vocab[idx])
            else:
                tokens.append(f"UNK-{idx}")
                
        print(f"{t:4d} | {tokens[0]:5s} ({top_values[0]:.2f}) | {tokens[1]:5s} ({top_values[1]:.2f}) | {tokens[2]:5s} ({top_values[2]:.2f})")
    
    # Simple greedy decoding with various blank penalties
    print("\nUsing Simple Greedy Decoding:")
    for blank_penalty in [0, 5, 10, 20, 50, 100]:
        # Apply blank penalty
        modified_logits = logits.copy()
        modified_logits[0, :, 2] -= blank_penalty
        
        # Decode using simple greedy approach
        decoded_tokens = decode_greedy(modified_logits[0])
        
        # Convert to text
        if decoded_tokens:
            pred_text = tokenizer.decode(decoded_tokens)
            print(f"With blank penalty {blank_penalty}: '{pred_text}'")
        else:
            print(f"With blank penalty {blank_penalty}: [empty]")
    
    # Show the true text
    true_indices = [idx for idx in true_label.numpy() if idx > 0]
    true_text = tokenizer.decode(true_indices)
    print(f"\nTrue text: '{true_text}'")

if __name__ == "__main__":
    test_single_example()