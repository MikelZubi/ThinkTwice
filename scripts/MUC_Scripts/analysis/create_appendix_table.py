import pandas as pd

# Read the original CSV
input_path = "/home/mzubillaga/DocIE/results/MUC/zeroshot/test/multilingual_analysis.csv"
output_path = "/home/mzubillaga/DocIE/results/MUC/zeroshot/test/appendix_table.csv"

df = pd.read_csv(input_path)

# Define the desired order for models and strategies
model_order = [
    "Llama3.3-70B", "LlamaR1-70B", "Qwen3-32B_nothink", "Qwen3-32B_think",
    "MUCLLAMA", "MUCR1", "MUCQWEN_nothink", "MUCQWEN_think"
]

strategy_order = ["greedy", "random_mean_score", "voter_majority", "voter_f1", "reward", "max_score"]

# Get unique languages in order of appearance
languages = df['language'].unique()

# Prepare the output data - rows are models + strategies, columns are languages
output_rows = []

for model in model_order:
    for strategy in strategy_order:
        row = {"model": model, "strategy": strategy}
        
        for lang in languages:
            lang_df = df[df['language'] == lang]
            model_data = lang_df[lang_df['modelname'] == model]
            if not model_data.empty:
                row[lang] = model_data[strategy].values[0]
            else:
                row[lang] = None
        
        output_rows.append(row)

# Create the output dataframe
output_df = pd.DataFrame(output_rows)

# Reorder columns: model, strategy, then languages
column_order = ["model", "strategy"] + list(languages)
output_df = output_df[column_order]

# Save to CSV
output_df.to_csv(output_path, index=False)

print(f"Appendix table saved to: {output_path}")
print(f"\nShape: {output_df.shape}")
print(f"\nFirst few rows:")
print(output_df.head(12))
