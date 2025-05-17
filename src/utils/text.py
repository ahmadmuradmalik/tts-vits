import re
import torch

# Define the character set for LJSpeech
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "

def text_to_sequence(text):
    """Convert text to sequence of character indices"""
    # Convert to uppercase and remove any characters not in the character set
    text = text.upper()
    text = re.sub(f'[^{CHARACTERS}]', '', text)
    
    # Create character to index mapping
    char_to_idx = {char: i for i, char in enumerate(CHARACTERS)}
    
    # Convert text to sequence of indices
    sequence = [char_to_idx[char] for char in text]
    return sequence

def sequence_to_text(sequence):
    """Convert sequence of character indices back to text"""
    idx_to_char = {i: char for i, char in enumerate(CHARACTERS)}
    text = ''.join([idx_to_char[idx] for idx in sequence])
    return text

def get_vocab_size():
    """Get the size of the vocabulary"""
    return len(CHARACTERS) 