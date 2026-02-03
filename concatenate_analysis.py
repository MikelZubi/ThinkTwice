import pandas as pd
import os

# Read the MUC multilingual analysis
muc_df = pd.read_csv('results/MUC/zeroshot/test/multilingual_analysis.csv')

# Read the BETTER analysis
better_df = pd.read_csv('results/BETTER/zeroshot/test/en_string_analysis.csv')

# Rename 'language' column to 'Dataset/Language' for MUC
muc_df = muc_df.rename(columns={'language': 'Dataset/Language'})

# Add 'Dataset/Language' column for BETTER
better_df['Dataset/Language'] = 'BETTER'

# Reorder columns to have 'Dataset/Language' first
cols = ['Dataset/Language'] + [col for col in better_df.columns if col != 'Dataset/Language']
better_df = better_df[cols]

# Concatenate both dataframes
combined_df = pd.concat([muc_df, better_df], ignore_index=True)

# Ensure the output directory exists
os.makedirs('results', exist_ok=True)

# Save the combined dataframe
combined_df.to_csv('results/analysis_combined.csv', index=False)

print(f"Combined CSV saved to 'results/analysis_combined.csv'")
print(f"Total rows: {len(combined_df)}")