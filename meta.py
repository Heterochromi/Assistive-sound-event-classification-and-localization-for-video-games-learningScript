import pandas as pd

# Load the CSV file
train = pd.read_csv('train.csv')

# Show the head (first 5 rows)
print("Head of the dataset:")
print(train.head())

print("\n" + "="*50 + "\n")

# Show the tail (last 5 rows)
print("Tail of the dataset:")
print(train.tail())



print("\n" + "="*50 + "\n")
print("number of unique  labels:")
# Split labels by comma, flatten the list, and count unique labels
all_labels = []
for label_string in train['labels']:
    # Split by comma and strip whitespace
    labels = [label.strip() for label in label_string.split(',')]
    all_labels.extend(labels)

unique_labels = set(all_labels)
print(f"Total unique individual labels: {len(unique_labels)}")
print(f"All unique labels: {sorted(unique_labels)}")
