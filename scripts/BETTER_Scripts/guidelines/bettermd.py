import os

# Define the directory and the ordered list of guideline files
guidelines_dir = "Docs/BETTER/"
ordered_files = [
    #"start.md",
    "start_simp.md",
    #"start_simp_indv.md",
    #"span.md",
    #"event.md",
    "Protestplate.md",
    "Corruplate.md",
    "Terrorplate.md",
    "Epidemiplate.md",
    "Disasterplate.md",
    "Displacementplate.md"
]

output_file = "Docs/BETTER_simplified.md"

with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in ordered_files:
        filepath = os.path.join(guidelines_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")
        else:
            print(f"Warning: {filepath} does not exist.")