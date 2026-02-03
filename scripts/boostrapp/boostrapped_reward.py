import json
import argparse

def remove_errors(all_templates):
    return [template for template in all_templates if ["ERROR"]  != template and [["ERROR"]] != template and "ERROR" not in template]

def bostrapped_reward(reward_file_path: str, boostrapp_file_path: str):
    """
    Reads a JSON file containing reward data and writes the data to a new file.

    Args:
        reward_file_path (str): Path to the input JSON file containing reward data.
        boostrapp_file_path (str): Path to the output file where the reward data will be written.
    """
    # Read the reward data from the JSON file
    reward_data = []
    with open(reward_file_path, 'r') as reward_file:
        for line in reward_file:
            reward_data.append(json.loads(line)["score_dict"])

    boostrapp_data = []
    # Read the boostrapp file
    with open(boostrapp_file_path, 'r') as boostrapp_file:
        for line in boostrapp_file:
            boostrapp_data.append(json.loads(line))
    
    output_data = []
    for reward_line, boostrapp_line in zip(reward_data, boostrapp_data):
        best_template = []
        max_log = float("-inf")
        boostrapped_templates = remove_errors(boostrapp_line["pred_json"])
        for template in boostrapped_templates:
            str_template = json.dumps(template, ensure_ascii=False)
            score = reward_line[str_template]
            if score > max_log:
                max_log = score
                best_template = template
            
        new_entry = {
            "docid": boostrapp_line["docid"],
            "doctext": boostrapp_line["doctext"],
            "pred_json_reward": best_template,
        }
        output_data.append(new_entry)
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process reward and boostrapp files.")
    parser.add_argument("--reward_file", type=str, required=True, help="Path to the reward JSON file.")
    parser.add_argument("--boostrapp_file", type=str, required=True, help="Path to the boostrapp JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    output_data = bostrapped_reward(args.reward_file, args.boostrapp_file)

    # Write the output data to the specified file
    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        for entry in output_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            output_file.write(json_line + '\n')
    print(f"Results written to {args.output_file}")