# Define the file paths
file1_path = 'labels/unigram1000_units.txt'  # Replace with your actual file path
file2_path = 'spm/unigram/unigram1000_units.txt'  # Replace with your actual file path

# Load the vocabularies from the files
def load_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}
    return vocab

# Load the vocabularies
vocab1 = load_vocab(file1_path)
vocab2 = load_vocab(file2_path)

# Find differences
unique_to_file1 = {token: count for token, count in vocab1.items() if token not in vocab2}
unique_to_file2 = {token: count for token, count in vocab2.items() if token not in vocab1}

# Output results
print(f"Tokens unique to file 1: {len(unique_to_file1)}")
for token, count in unique_to_file1.items():
    print(f"{token}: {count}")

print(f"\nTokens unique to file 2: {len(unique_to_file2)}")
for token, count in unique_to_file2.items():
    print(f"{token}: {count}")
