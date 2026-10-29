
import re

sentence = "I REALLY love this product!!!"

# 1. Convert to lowercase
text = sentence.lower()

# 2. Remove punctuation
text = re.sub(r"[^\w\s]", "", text)

# 3. Tokenization
tokens = text.split()

# 4. Remove basic stop words
stop_words = {"i", "this", "is", "the", "a", "an"}

filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]

print("Original:", sentence)
print("Lowercase:", sentence.lower())
print("Cleaned text:", text)
print("Tokens:", tokens)
print("After stop-word removal:", filtered_tokens)
