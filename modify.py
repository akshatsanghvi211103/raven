input_file = "spm/unigram/unigram1000_units.txt"  # Replace with your input file name
output_file = "spm/unigram/unigram1000_units_modified.txt"  # Replace with your desired output file name

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Split line into token and count
        parts = line.split()
        if len(parts) < 2:
            continue  # Skip malformed lines

        token, count = parts[0], parts[1]

        # Check if the token starts with ▁ or not
        if not token.startswith('▁') and token[0].isalpha():
            token = '▁' + token  # Add ▁ to the start

        # Write the modified token and count to the output file
        outfile.write(f"{token} {count}\n")

print(f"Modified unigram file saved to: {output_file}")
