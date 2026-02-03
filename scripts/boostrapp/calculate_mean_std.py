import pandas as pd 
import os
import argparse


def calculate_mean_std(input_files,output_path,BETTER=False):
    all_results = []
    for file in input_files:
        df = pd.read_csv(file,header=0).T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        all_results.append(df)
    combined_df = pd.concat(all_results)
    combined_df_mean = combined_df.groupby(combined_df.index).mean()
    combined_df_std = combined_df.groupby(combined_df.index).std()
    combined_df = (combined_df_mean.astype(str) + "±" + combined_df_std.astype(str)).T
    print(combined_df)
    combined_df.to_csv(output_path)

def calculate_through_paths(n=3, paths=["results/BETTER/zeroshot/test/en_string",
                                 "results/MUC/zeroshot/test/en",
                                 "results/MUC/zeroshot/test/ar",
                                 "results/MUC/zeroshot/test/fa",
                                 "results/MUC/zeroshot/test/ko",
                                 "results/MUC/zeroshot/test/ru",
                                 "results/MUC/zeroshot/test/zh"]):
    for path in paths:
        voterf1_files = []
        votermajority_files = []
        reward_files = []
        for i in range(n):
            boostrapp_path = os.path.join(path,f"boostrapp{i}")
            if "BETTER" in path:
                voterf1_file = os.path.join(boostrapp_path, "voterf1_scores.csv")
                voterf1_files.append(voterf1_file)
                votermajority_file = os.path.join(boostrapp_path, "voterMajority_scores.csv")
                votermajority_files.append(votermajority_file)
            else:
                voterf1_file = os.path.join(boostrapp_path, "voterf1.csv")
                voterf1_files.append(voterf1_file)
                votermajority_file = os.path.join(boostrapp_path, "voterMajority.csv")
                votermajority_files.append(votermajority_file)
                reward_file = os.path.join(boostrapp_path, "Reward.csv")
                reward_files.append(reward_file)
        calculate_mean_std(voterf1_files, os.path.join(path,"voterf1_mean_std.csv"), BETTER="BETTER" in path)
        calculate_mean_std(votermajority_files, os.path.join(path,"votermajority_mean_std.csv"), BETTER="BETTER" in path)
        if len(reward_files)>0:
            calculate_mean_std(reward_files, os.path.join(path,"reward_mean_std.csv"), BETTER=False)
    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--n", type=int, default=3, help="Number of boostrapp samples created")
    args = argument_parser.parse_args()
    n = args.n
    calculate_through_paths(n=n)