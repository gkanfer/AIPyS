import pandas as pd
import os
path = r'D:\Gil\images\pex_project\outproc_output\022224'
df = pd.read_csv(os.path.join(path,"filelist.csv"), header=None, names=['FileName', 'CreationTime'])
# Filter to keep only rows where the file name ends with '.png'
df = df[df['FileName'].str.endswith('.png')]
# Convert CreationTime from string to datetime
df['CreationTime'] = pd.to_datetime(df['CreationTime'])
df['CreationTimeMinute'] = df['CreationTime'].dt.floor('T')

# Group by the truncated CreationTime and count the occurrences
files_per_minute = df.groupby('CreationTimeMinute').size()

# If you need to reset the index to have CreationTimeMinute as a column
files_per_minute = files_per_minute.reset_index(name='FileCount')

# Optionally, save the result to a new CSV file
files_per_minute.to_csv('files_per_minute.csv', index=False)

print(files_per_minute)