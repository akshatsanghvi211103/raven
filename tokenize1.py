import sentencepiece as spm
import random

# Initialize SentencePiece processor
sp = spm.SentencePieceProcessor()
sp.load("spm/unigram/unigram5000.model")  # Path to your SentencePiece model

# Input and output files
input_file = "/ssd_scratch/cvit/akshat/datasets/accented_speakers/the_book_leo/all_labels.txt"  # Replace with your input file path
output_file = "output.txt"  # Replace with your output file path

# Function to extract metadata from the input line
def extract_metadata_and_sentence(line):
    # Split into path and sentence
    path, sentence = line.split(" ", 1)
    folder_name, video_name = path.split("/", 1)
    folder_name = "the_book_leo"
    return folder_name, video_name, sentence

# Process the input file
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        # Extract metadata and sentence
        folder_name, video_name, sentence = extract_metadata_and_sentence(line)

        # Generate a random integer (you can set the range as needed)
        random_integer = random.randint(10, 99)

        # Tokenize the sentence into numeric IDs
        token_ids = sp.EncodeAsIds(sentence)
        token_id_str = " ".join(map(str, token_ids))  # Join IDs into a single string

        # Write the output in the desired format
        fout.write(f"{folder_name},{video_name},{random_integer},{token_id_str}\n")
