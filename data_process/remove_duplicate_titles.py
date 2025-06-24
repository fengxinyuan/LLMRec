import json
from collections import OrderedDict

def remove_duplicates_by_title(input_path, output_path):
    """
    Reads a JSON file, removes duplicate entries based on the 'title' field,
    and writes the unique entries back to a new file.

    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path to the output JSON file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        unique_titles = set()
        unique_data = []
        
        for item in data:
            title = item.get('title')
            if title and title not in unique_titles:
                unique_titles.add(title)
                unique_data.append(item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
            
        print(f"Removed {len(data) - len(unique_data)} duplicate(s).")
        print(f"Processed data saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {input_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    input_file = 'data/renamed_knowledge_processed.json'
    output_file = 'data/cleared_knowledge_processed.json' # Overwrite the original file
    remove_duplicates_by_title(input_file, output_file)