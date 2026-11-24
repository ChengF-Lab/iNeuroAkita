#!/usr/bin/env python
"""check_tfrecord.py -- print the shape and the first values of one training batch of an akita_data.py dataset

Usage: python check_tfrecord.py <akita_data output dir, e.g. model_input/fold1>
"""
import sys
import tensorflow as tf
from basenji import dataset

data_dir = sys.argv[1]


# Create a dataset object
seq_data = dataset.SeqDataset(
    data_dir, 
    split_label="train",
    batch_size=1, 
    mode="eval"
)

# Print dataset structure
print(seq_data)

# Take a single batch from the dataset
for batch in seq_data.dataset.take(1):
    seq_input, target_output = batch
    print("Sequence Shape:", seq_input.shape)
    print("Target Shape:", target_output.shape)


# Take a single batch from the dataset
for batch in seq_data.dataset.take(1):
    seq_input, target_output = batch
    # Convert tensors to NumPy arrays and print actual data
    print("Example Sequence Data (first 10 elements):")
    #print(seq_input.numpy().flatten()[:10])  # Print first 10 values for readability
    print(seq_input.numpy()[:10])  # Print first 10 values for readability
    print("\nExample Target Data (first 10 elements):")
   # print(target_output.numpy().flatten()[:10])  # Print first 10 values for readability
    print(target_output.numpy()[:10])
    # Print full arrays if needed (WARNING: Can be very large!)
    # print("\nFull Sequence Data:", seq_input.numpy())
    # print("\nFull Target Data:", target_output.numpy())
    break  # Stop after one batch
