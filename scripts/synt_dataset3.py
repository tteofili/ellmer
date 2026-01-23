import pandas as pd
from faker import Faker
import random
import string

# Initialize Faker
fake = Faker()
random.seed(42)  # For reproducibility


# Function to introduce variations in textual attributes
def introduce_variation(text, variation_level='medium'):
    """
    Introduce realistic variations to simulate data inconsistencies.
    variation_level: 'low', 'medium', 'high' to control noise.
    """
    if random.random() < 0.05:  # Low chance to leave unchanged
        return text

    variations = []
    if variation_level == 'high':
        prob = 0.8
    elif variation_level == 'medium':
        prob = 0.5
    else:
        prob = 0.2

    words = text.split()
    new_words = []
    for i, word in enumerate(words):
        if random.random() < prob:
            # Typo: Replace a random char
            if len(word) > 1 and random.random() < 0.4:
                idx = random.randint(0, len(word) - 1)
                word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:]

            # Abbreviation or shortening
            elif random.random() < 0.3:
                if len(word) > 3:
                    word = word[:3] + '.' if random.choice([0, 1]) else word[:random.randint(1, len(word) - 2)]
                else:
                    word = word[0] + '.'

            # Synonym replacement (simplified common ones)
            elif random.random() < 0.2:
                synonyms = {
                    'street': ['st', 'strt'], 'road': ['rd', 'rd.'], 'company': ['co', 'corp'],
                    'university': ['uni', 'u.'], 'institute': ['inst'], 'center': ['ctr']
                }
                for key, vals in synonyms.items():
                    if key in word.lower():
                        word = random.choice(vals)
                        break

            # Missing or extra char
            elif random.random() < 0.1:
                if len(word) > 1:
                    word = word[:random.randint(0, len(word) - 1)]
                else:
                    word += random.choice(string.ascii_lowercase)

        new_words.append(word)

    new_text = ' '.join(new_words)

    # Overall changes: Swap words, add/remove punctuation
    if random.random() < 0.1:
        if len(new_words) > 1:
            i, j = random.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        new_text = ' '.join(new_words)

    return new_text.strip()


# Function to generate base attributes for an entity
def generate_base_entity(num_attributes=100):
    entity = {}
    for i in range(num_attributes):
        attr_name = f'attribute_{i + 1}'
        # Vary the type of fake data
        attr_types = ['name', 'address', 'company', 'sentence', 'text']
        typ = random.choice(attr_types)
        if typ == 'name':
            entity[attr_name] = fake.name()
        elif typ == 'address':
            entity[attr_name] = fake.address().replace('\n', ', ')
        elif typ == 'company':
            entity[attr_name] = fake.company()
        elif typ == 'sentence':
            entity[attr_name] = fake.sentence(nb_words=random.randint(3, 8))
        else:  # 'text'
            entity[attr_name] = ' '.join([fake.word() for _ in range(random.randint(2, 5))])
    return entity


# Parameters
num_shared_entities = 150  # Entities that appear in both sources (with matches)
num_a_only = 150  # Unique to source A
num_b_only = 150  # Unique to source B
num_attributes = 105  # At least 100 textual attributes
variation_a = 'medium'  # Variation level for source A
variation_b = 'high'  # Slightly higher for source B to simulate differences

# Step 1: Generate entities
print("Generating base entities...")
shared_entities = [generate_base_entity(num_attributes) for _ in range(num_shared_entities)]
a_only_entities = [generate_base_entity(num_attributes) for _ in range(num_a_only)]
b_only_entities = [generate_base_entity(num_attributes) for _ in range(num_b_only)]

# Step 2: Generate Source A records
print("Generating Source A...")
source_a_records = []
id_a_counter = 1
entity_to_id_a = {}  # Map shared entity index to its ID in A

# Shared entities in A
for idx, entity in enumerate(shared_entities):
    record = {attr: introduce_variation(val, variation_a) for attr, val in entity.items()}
    record['id'] = id_a_counter
    source_a_records.append(record)
    entity_to_id_a[idx] = id_a_counter
    id_a_counter += 1

# A-only entities
for entity in a_only_entities:
    record = {attr: introduce_variation(val, variation_a) for attr, val in entity.items()}
    record['id'] = id_a_counter
    source_a_records.append(record)
    id_a_counter += 1

source_a_df = pd.DataFrame(source_a_records)
source_a_df = source_a_df[['id'] + [col for col in source_a_df.columns if col != 'id']]  # ID first
source_a_df.to_csv('source_a.csv', index=False)
print(f"Source A: {len(source_a_df)} records, {len(source_a_df.columns) - 1} attributes saved.")

# Step 3: Generate Source B records
print("Generating Source B...")
source_b_records = []
id_b_counter = 1
entity_to_id_b = {}  # Map shared entity index to its ID in B

# Shared entities in B
for idx, entity in enumerate(shared_entities):
    record = {attr: introduce_variation(val, variation_b) for attr, val in entity.items()}
    record['id'] = id_b_counter
    source_b_records.append(record)
    entity_to_id_b[idx] = id_b_counter
    id_b_counter += 1

# B-only entities
for entity in b_only_entities:
    record = {attr: introduce_variation(val, variation_b) for attr, val in entity.items()}
    record['id'] = id_b_counter
    source_b_records.append(record)
    id_b_counter += 1

source_b_df = pd.DataFrame(source_b_records)
source_b_df = source_b_df[['id'] + [col for col in source_b_df.columns if col != 'id']]  # ID first
source_b_df.to_csv('source_b.csv', index=False)
print(f"Source B: {len(source_b_df)} records, {len(source_b_df.columns) - 1} attributes saved.")

# Step 4: Generate Ground Truth
print("Generating Ground Truth...")
ground_truth = []
# Only the shared entities are matches (1:1 for simplicity; in real ER, could have 1:many)
for idx in range(num_shared_entities):
    pair = {
        'id_a': entity_to_id_a[idx],
        'id_b': entity_to_id_b[idx],
        'label': 1  # Match
    }
    ground_truth.append(pair)

# Optionally, add some negative samples for evaluation (random non-matches)
num_negatives = 200  # Sample of negative pairs
all_possible_a_shared = list(entity_to_id_a.values())
all_possible_b_shared = list(entity_to_id_b.values())
for _ in range(num_negatives):
    id_a = random.choice(all_possible_a_shared)
    id_b = random.choice(all_possible_b_shared)
    # Ensure it's not a true match
    while entity_to_id_b[list(entity_to_id_a.keys())[list(entity_to_id_a.values()).index(id_a)]] == id_b:
        id_b = random.choice(all_possible_b_shared)
    pair = {'id_a': id_a, 'id_b': id_b, 'label': 0}  # Non-match
    ground_truth.append(pair)

ground_truth_df = pd.DataFrame(ground_truth)
ground_truth_df.to_csv('ground_truth.csv', index=False)
print(f"Ground Truth: {len(ground_truth_df)} pairs ({num_shared_entities} positives, {num_negatives} negatives) saved.")

print("\nDataset generated successfully!")
print("- source_a.csv: 300 records (150 shared + 150 unique), 105 textual attributes")
print("- source_b.csv: 300 records (150 shared + 150 unique), 105 textual attributes")
print("- ground_truth.csv: Matching and sample non-matching pairs between sources")
print("\nTo use: Run this script in a Python environment with pandas and faker installed.")
print("pip install pandas faker")