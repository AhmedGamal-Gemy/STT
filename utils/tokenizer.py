import tensorflow as tf
from typing import List,Dict
import matplotlib.pyplot as plt
import numpy as np

class Tokenizer:

    def __init__(self, vocab: List[str] = None, max_text_length: int = 100):

            self.special_tokens = {"<pad>": 0, "<unk>": 1, "<blank>": 2}  
            self.vocab = list(self.special_tokens.keys())  # Initialize with special tokens
            self.max_text_length = max_text_length
            self.char_to_id = {}
            self.id_to_char = {}
            self.chars_count = {}

    # Get all vocablary from the dataset and create mappings from and to char
    def build_vocab(self, dataset: tf.data.Dataset) -> None:
        self.char_counts = {}  # Reset counts when building vocab
        
        # Debug the dataset structure
        sample_batch = next(iter(dataset))
        print("DEBUG: Dataset sample structure:")
        print(f"Keys in batch: {list(sample_batch.keys())}")
        
        for batch in dataset:
            try:
                # Try to access the sentence field
                if 'sentence' in batch:
                    text_field = batch['sentence']
                else:
                    # Print all keys to debug
                    print(f"Warning: 'sentence' not in batch. Available keys: {list(batch.keys())}")
                    # Try alternate fields
                    if 'text' in batch:
                        text_field = batch['text']
                    elif 'transcript' in batch:
                        text_field = batch['transcript']
                    else:
                        print("Could not find text field in batch")
                        continue
                        
                # Process the text
                if isinstance(text_field, bytes):
                    text_batch = text_field.decode('utf-8').lower()
                else:
                    text_batch = str(text_field).lower()
                    
                # Count characters
                for char in text_batch:
                    self.char_counts[char] = self.char_counts.get(char, 0) + 1
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
        
        chars = sorted(self.char_counts.keys())
        self.vocab = list(self.special_tokens.keys()) + chars
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    # Encode from char to indexes and return tensor
    def encode(self, text: str) -> tf.Tensor:
        text = text.lower()[:self.max_text_length]  # Truncate long texts
        
        # Iterating through each char in text and get its coressponding index from the existing vocab
        # And if it didn't find the char in the existing vocab return the special unkown token
        ids = [ self.char_to_id.get( char, self.special_tokens['<unk>'] ) for char in text  ]

        return tf.convert_to_tensor(ids, dtype = tf.int32)

    # Decode from indexes to char and return string
    def decode(self, token_ids):

        # Handle different input types
        if hasattr(token_ids, 'numpy'):
            token_ids = token_ids.numpy()  # Convert TF tensor to numpy array
        
        # Filter out padding (0) and very large indices
        token_ids = [id for id in token_ids if id > 0 and id < len(self.vocab)]
        
        # If empty, return special string
        if not token_ids:
            return "[empty]"
            
        # Convert IDs to characters and join
        try:
            return ''.join([self.vocab[id] for id in token_ids])
        except IndexError:
            # Handle any out-of-range indices
            valid_ids = [id for id in token_ids if 0 <= id < len(self.vocab)]
            return ''.join([self.vocab[id] for id in valid_ids])
    

    # Some visualizations
    def visualize_vocab(self, top_k: int = 20) -> None:

        # Exclude special tokens and sort by frequency
        sorted_chars = sorted(
            self.char_counts.items(), 
            key=lambda x: -x[1]
        )[:top_k]
        
        if not sorted_chars:
            print("No character frequency data available.")
            return
        
        chars, counts = zip(*sorted_chars)
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(chars)), counts)
        plt.xticks(range(len(chars)), chars)
        plt.title(f"Top {top_k} Frequent Characters")
        plt.show()


# Now testing the tokenizer on mock data

if __name__ == "__main__":
    
    mock_texts = [
        "Hello worldddddddddd",
        "This is a test",
        "Speech recognition"
    ]

    mock_dataset = tf.data.Dataset.from_tensor_slices( {
        "sentence" : [ text.encode('utf8') for text in mock_texts ]
    } )

    tokenizer = Tokenizer()

    tokenizer.build_vocab(mock_dataset)


    test_text = "hello from test ?"

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded.numpy()}")
    print(f"Decoded: {decoded}")

    tokenizer.visualize_vocab(5)