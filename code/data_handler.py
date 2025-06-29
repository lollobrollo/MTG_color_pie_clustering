import requests
import shutil
import ijson
import json
import gzip
import decimal
import os
import tempfile

def download_data(output_path):
    """
    Downloads the latest version of Scryfall's 'default_cards' bulk data.
    - Fetches metadata from the Scryfall bulk API
    - Extracts the download URL for the 'default_cards' dataset
    - Streams and decompresses the .json.gz file directly to disk
    - Saves the uncompressed JSON to the specified output path
    """
    print("Fetching latest bulk metadata...")
    bulk_api_url = "https://api.scryfall.com/bulk-data"

    response = requests.get(bulk_api_url)
    response.raise_for_status()

    bulk_data = response.json()["data"]
    
    # Find the 'default_cards' entry
    default_entry = next((entry for entry in bulk_data if entry["type"] == "default_cards"), None)
    if not default_entry:
        raise ValueError("Couldn't find 'default_cards' in Scryfall bulk data.")

    download_url = default_entry["download_uri"]
    print(f"Downloading latest default-cards from:\n{download_url}")

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        # Decompress while streaming directly to a JSON file
        with gzip.GzipFile(fileobj=response.raw) as gzipped:
            with open(output_path, 'wb') as out_file:
                shutil.copyfileobj(gzipped, out_file)
                
    print(f"Download complete. Saved to '{output_path}'")


def get_all_fields(input_file):
    """ Scans the input JSON file and returns a set of all unique field names used across cards """
    fields = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, "item"):
            fields.update(obj.keys())
    return fields


def filter_data_by_field(input_file, output_file, json_path='item'):
    """
    Filters a JSON file, keeping only specific fields from each card.
    - Outputs the filtered result to a new file as a JSON array
    - Converts `decimal.Decimal` values to float for JSON compatibility
    - Returns the set of fields that were retained
    """

    keep_fields = {
    "oracle_text", "type_line", "keywords", "name", "mana_cost", "colors",
    "color_identity", "games", "layout", "card_faces", "color_indicator",
    "hand_modifier", "life_modifier", "oracle_id", "released_at", "set", "set_name"
    }

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        
        outfile.write("[\n")  # start JSON array
        first = True
        
        for card in ijson.items(infile, json_path):
            filtered_card = {k: v for k, v in card.items() if k in keep_fields}
            
            if not first:
                outfile.write(",\n")
            else:
                first = False
            
            json.dump(filtered_card, outfile, ensure_ascii=False,
                    default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))
        
        outfile.write("\n]")  # end JSON array

    print(f"Filtered cards saved to: {output_file}")
    return keep_fields


def filter_data_by_relevance(input_file, json_path='item'):
    """
    Filters the dataset in-place to retain only 'relevant' cards for semantic analysis.
    Relevance rules:
    - Excludes Basic Lands
    - Excludes tokens
    - Excludes cards without color identity
    - Excludes cards with no rules text and zero CMC
    - Excludes cards not printed in paper
    Output:
    - Writes to a temporary file first
    - Then overwrites the original file with the filtered version
    """
    def is_relevant(card):
        if "Basic Land" in card.get("type_line", ""):
            return False
        if card.get("layout") == "token":
            return False
        if len(card.get("color_identity", [])) == 0:
            return False  # optional, see reasoning above
        if card.get("oracle_text", "").strip() == "" and card.get("cmc", 0) == 0:
            return False  # mostly irrelevant lands
        if "paper" not in card.get("games", []):
            return False # chose to focus on paper format
        return True
    
    # Create a temporary file in the same directory
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(input_file))
    os.close(temp_fd)  # We'll open it properly later

    with open(input_file, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
        
        outfile.write("[\n")
        first = True

        for card in ijson.items(infile, json_path):
            if is_relevant(card):
                if not first:
                    outfile.write(",\n")
                else:
                    first = False
                json.dump(card, outfile, ensure_ascii=False,
                    default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))
        
        outfile.write("\n]")

    # Replace original file with filtered version
    shutil.move(temp_path, input_file)
    print(f"Filtered data saved to: {input_file}")



if __name__ == "__main__":
    
    download = input("Download fresher data? (Y/N): ").strip().lower() == "y"
    raw_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_cards.json"))
    clean_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered_cards.json"))

    if download:
        download_data(raw_data)

    print("Applying first filter, based on fields of .json file...")
    fields = filter_data_by_field(raw_data, clean_data)
    print("Applying second filter, based on relevance of cards...")
    filter_data_by_relevance(clean_data)