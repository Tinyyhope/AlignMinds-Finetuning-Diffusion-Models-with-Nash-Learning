# Import Packages
import pandas as pd
import random
import json

# Parameters
input_file = '/Users/tinyhope/Desktop/PartiPrompts.tsv'  
output_file = '/Users/tinyhope/Desktop/filtered_prompts.json'  # Output path for the JSON file
min_word_count = 10  # Minimum number of words required in a valid prompt
forbidden_phrases = ['in the style of']  # List of phrases that disqualify a prompt
max_prompts = 1000  # Maximum number of prompts to save

# Load the input file as a DataFrame
df = pd.read_csv(input_file, sep='\t')

# Define prompt filtering rules
def is_valid_prompt(row):
    """
    Determines whether a prompt is valid based on a set of filtering rules:
    1. Exclude prompts where Category is 'Abstract' and Challenge is 'Basic'.
    2. Exclude prompts that contain any forbidden phrase.
    3. Exclude prompts with word count less than `min_word_count`.

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'Prompt', 'Category', and 'Challenge'.

    Returns:
        bool: True if the prompt satisfies all criteria; False otherwise.
    """
    prompt = str(row['Prompt']).strip()
    category = str(row['Category']).strip()
    challenge = str(row['Challenge']).strip()
    lower_prompt = prompt.lower()

    if category == 'Abstract' and challenge == 'Basic':
        return False
    if any(phrase in lower_prompt for phrase in forbidden_phrases):
        return False
    if len(prompt.split()) < min_word_count:
        return False

    return True

# Apply the validation function to each row to filter prompts
filtered = df[df.apply(is_valid_prompt, axis=1)]

# Remove prompts that begin with the same first 30 characters (case-insensitive)
def deduplicate(prompts):
    """
    Remove prompts that begin with the same first 30 characters (case-insensitive),
    ensuring uniqueness in phrasing.

    Parameters:
        prompts (list of str): List of valid prompt strings.

    Returns:
        list of str: Deduplicated list of prompts.
    """
    seen = set()
    unique_prompts = []
    for p in prompts:
        key = p[:30].lower()
        if key not in seen:
            seen.add(key)
            unique_prompts.append(p)
    return unique_prompts

# Extract and clean the prompt list
prompt_list = filtered['Prompt'].dropna().tolist()
prompt_list = deduplicate(prompt_list)

# Shuffle prompts to randomize order
random.shuffle(prompt_list)

# Keep only the first `max_prompts` entries
final_prompts = prompt_list[:max_prompts]

# Format the data for JSON output
json_data = [{"prompt": p.strip()} for p in final_prompts]

# Write the final result to a JSON file with UTF-8 encoding
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Prompt filtering completed successfully. A total of {len(json_data)} prompts were saved to: {output_file}")

