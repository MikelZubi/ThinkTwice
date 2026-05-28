import pandas as pd

# Read the original CSV
input_path = "results/MUC/zeroshot/test/multilingual_analysis.csv"
output_path = "results/MUC/zeroshot/test/appendix_table.csv"

df = pd.read_csv(input_path)

# Define the desired order for models and strategies
model_order = [
    "Llama3.3-70B", "LlamaR1-70B", "Qwen3-32B_nothink", "Qwen3-32B_think",
    "MUCR1", "MUCQWEN_think"
]

strategy_order = ["greedy", "voter_majority_mean_std", "voter_f1_mean_std", "reward_mean_std", "max_score"]

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

# Calculate the average over all languages for each row
def calculate_average(row, langs):
    vals = []
    stds = []
    has_std = False
    for lang in langs:
        val = row[lang]
        if pd.isna(val):
            continue
        if isinstance(val, str) and '±' in val:
            parts = val.split('±')
            vals.append(float(parts[0]))
            stds.append(float(parts[1]))
            has_std = True
        else:
            vals.append(float(val))
            
    if not vals:
        return None
        
    mean_val = sum(vals) / len(vals)
    
    if has_std and stds:
        # Convert std to variance, calculate mean of variances, then sqrt back to std
        mean_var = sum(s ** 2 for s in stds) / len(stds)
        avg_std = mean_var ** 0.5
        return f"{mean_val}±{avg_std}"
    else:
        return mean_val

output_df['Average'] = output_df.apply(lambda row: calculate_average(row, languages), axis=1)

# Format values to 2 decimal places (handling both plain numbers and strings with ±)
def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, str) and '±' in val:
        parts = val.split('±')
        return rf"{float(parts[0]):.2f}${{\scriptscriptstyle\pm}}$\tiny{{{float(parts[1]):.2f}}}"
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return val

for col in list(languages) + ['Average']:
    output_df[col] = output_df[col].apply(format_value)

# Reorder columns: model, strategy, then languages, then Average
column_order = ["model", "strategy"] + list(languages) + ["Average"]
output_df = output_df[column_order]

# Save to CSV
output_df.to_csv(output_path, index=False)

print(f"Appendix table saved to: {output_path}")
print(f"\nShape: {output_df.shape}")
print(f"\nFirst few rows:")
print(output_df.head(12))
