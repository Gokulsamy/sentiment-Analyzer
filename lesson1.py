
sentence = "I really love this product!"

# Step 1: Convert text to lowercase
clean_text = sentence.lower()

# Step 2: Split sentence into words
tokens = clean_text.split()

print("Original sentence:", sentence)
print("Lowercase text:", clean_text)
print("Tokens:", tokens)
print("Number of words:", len(tokens))
