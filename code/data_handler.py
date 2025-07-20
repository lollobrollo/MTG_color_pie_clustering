import requests
import shutil
import ijson
import json
import gzip
import decimal
import os
import tempfile
import pandas as pd
from datetime import datetime
import re

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
                
    print(f"Download complete.")


def get_all_fields(input_file):
    """ Scans the input JSON file and returns a set of all unique field names used across cards """
    fields = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, "item"):
            fields.update(obj.keys())
    return fields


def filter_json_fields(input_file, fields_to_keep, output_file=None, inplace=False):
    """
    Filters a JSON file by keeping only the specified fields in each object.

    Args:
        - input_file (str) : Path to the input JSON file.
        - fields_to_keep (set) : Fields to retain in each JSON object.
        - output_file (str) : Output path (ignored if inplace=True).
        - inplace (bool) : If True, overwrite the input file with filtered data.
    """

    if inplace:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(input_file))
        os.close(temp_fd)
        output_path = temp_path
    else:
        output_path = output_file
    assert output_path, "Ziopera non esiste"

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("[\n")
        first = True

        for card in ijson.items(infile, 'item'):
            filtered = {k: v for k, v in card.items() if k in fields_to_keep}
            
            if not first:
                outfile.write(",\n")
            else:
                first = False

            json.dump(filtered, outfile, ensure_ascii=False,
                      default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))

        outfile.write("\n]")

    if inplace:
        shutil.move(output_path, input_file)


def filter_data_by_relevance(input_file, json_path='item'):
    """
    Filters the dataset in-place to retain only 'relevant' cards for semantic analysis.
    Also merges multi-faced cards into a single card for later use.
    Relevance rules:
    - Excludes Basic Lands
    - Excludes tokens
    - Excludes cards with no rules text and zero CMC
    - Excludes cards not printed in paper
    - Excludes non-english cards
    - Excludes cards not meant for structured play (mostly the un-sets)
    Output:
    - Writes to a temporary file first
    - Then overwrites the original file with the filtered version
    """
    def is_relevant(card):
        if "Basic Land" in card.get("type_line", ""):
            return False
        if card.get("layout") == "token":
            return False
        if card.get("oracle_text", "").strip() == "" and card.get("cmc", 0) == 0:
            return False  # mostly irrelevant lands
        if "paper" not in card.get("games", []):
            return False # chose to focus on paper format
        if card.get("lang", "en") != "en":
            return False  # exclude non-English cards
        # Exclude cards not intended for structured play
        if card.get("border_color") == "silver":
            return False
        if card.get("security_stamp") == "acorn":
            return False
        if card.get("set_type") == "funny":
            return False
        if all(v == "not_legal" for v in card.get("legalities", {}).values()):
            return False
        return True
    
    seen_cards = set()  # Store (oracle_id, released_at) pairs

    # Create a temporary file in the same directory
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(input_file))
    os.close(temp_fd)  # We'll open it properly later

    with open(input_file, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
        
        outfile.write("[\n")
        first = True

        for card in ijson.items(infile, json_path):

            # if handling a multi-faced card, merge fields and bring them to top-level
            if not card.get("oracle_text") and isinstance(card.get("card_faces"), list):
                faces = card["card_faces"]
                card["oracle_text"] = "\n".join(f.get("oracle_text", "") for f in faces if f.get("oracle_text"))
                card["type_line"] = " // ".join(f.get("type_line", "") for f in faces if f.get("type_line"))
                card["name"] = " // ".join(f.get("name", "") for f in faces if f.get("name"))
                card["mana_cost"] = " // ".join(f.get("mana_cost", "") for f in faces if f.get("mana_cost"))
                all_keywords = set()
                for f in faces:
                    all_keywords.update(f.get("keywords", []))
                card["keywords"] = list(all_keywords)

            if not is_relevant(card):
                continue
            
            # Keep one copy per printing date (e.g. ignore foils) 
            key = (card.get("oracle_id"), card.get("released_at"))
            if key in seen_cards:
                continue
            seen_cards.add(key)

            # If the card is colorless, update its color identity
            if len(card.get("color_identity", [])) == 0:
                card["color_identity"] = ["C"]

            if not first:
                outfile.write(",\n")
            else:
                first = False
            json.dump(card, outfile, ensure_ascii=False,
                default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))

        outfile.write("\n]")

    # Replace original file with filtered version
    shutil.move(temp_path, input_file)


def filter_data(raw_data, clean_data, show=False):
    """
    Function that cals all filters in one place and returns the file path of the cleaned data.
    Returns the fields that have been kept after remopving useless ones and the ones useful only in preprocessing.
    If 'show' == True, print the head of the cleaned dataset
    """

    print(f"Applying filters...")
    
    keep_fields = {
    "id", "oracle_text", "type_line", "name", "lang", "keywords", "mana_cost", "colors",
    "color_identity", "games", "layout", "card_faces", "color_indicator",
    "hand_modifier", "life_modifier", "oracle_id", "released_at", "set", "set_name",
    "border_color", "security_stamp", "set_type", "legalities"
    }
    filter_json_fields(raw_data, keep_fields, clean_data)

    filter_data_by_relevance(clean_data)

    keep_fields ={
    "id", "oracle_id", "name", "set", "set_name", "oracle_text",
    "type_line", "mana_cost", "colors", "color_identity",
    "keywords", "hand_modifier", "life_modifier", "released_at"
    }
    filter_json_fields(clean_data, keep_fields, inplace = True)

    print(f"Filtering completed.")
    return keep_fields


def get_monocolored_cards(input_file, output_file, json_path='item'):
    """
    Filters the cleaned dataset to include only monocolored cards (based on 'color_identity') and writes them to 'output_file'.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("[\n")
        first = True
        for card in ijson.items(infile, json_path):
            colors = card.get("colors", [])
            if isinstance(colors, list) and len(colors) == 1:
                if not first:
                    outfile.write(",\n")
                else:
                    first = False
                json.dump(card, outfile, ensure_ascii=False,
                          default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))
        outfile.write("\n]")


def count_cards(file_path):
    total = 0
    monocolored = 0
    multicolored = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Reading data...')
        for card in ijson.items(f, 'item'):
            total += 1
            colors = card.get("colors", [])
            if isinstance(colors, list):
                if len(colors) == 1:
                    monocolored += 1
                elif len(colors) > 1:
                    multicolored += 1
    print(f"Total cards: {total}")
    print(f"Monocolored cards: {monocolored} ({monocolored / total:.2%})")
    print(f"Multicolored cards: {multicolored} ({multicolored / total:.2%})\n")



if __name__ == "__main__":
    
    # download = input("Download fresher data? (Y/N): ").strip().lower() == "y"
    raw_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_cards.json"))
    clean_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "clean_cards.json"))
    monocolored_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "monocol.json"))

    # if download:
    #     download_data(raw_data)


    #filter_data(raw_data, clean_data)

    """
    count_cards(raw_data)
    count_cards(clean_data)
    count_cards(monocolored_data)

    Reading data...
    Total cards: 108565
    Monocolored cards: 68330 (62.94%)
    Multicolored cards: 13171 (12.13%)

    Reading data...
    Total cards: 65551
    Reading data...
    Total cards: 65551
    Monocolored cards: 45925 (70.06%)
    Multicolored cards: 7248 (11.06%)

    Reading data...
    Total cards: 45925
    Monocolored cards: 45925 (100.00%)
    Multicolored cards: 0 (0.00%)
    """

    # df = pd.read_parquet("hf://datasets/minimaxir/mtg-embeddings/mtg_embeddings.parquet")
    # df.head()
    # hf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.parquet"))
    # df.to_parquet(hf_path)

    
    # Add a column to the dataframe in data.parquet with all printing years associated to the cards (rows) 
    set_year_dict = {}
    with open(clean_data, 'r', encoding='utf-8') as f:
        for card in ijson.items(f, 'item'):
            set_name = card.get("set")
            released_at = card.get("released_at")
          
            if set_name and released_at:
                set_name = set_name.upper()
                year = datetime.strptime(released_at, "%Y-%m-%d").year
                if set_name not in set_year_dict:
                    set_year_dict[set_name] = year
                else:
                    set_year_dict[set_name] = max(set_year_dict[set_name], year)
    #print(f"Here's the dict: {set_year_dict}")
    
    parquet_old = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet", "data.parquet"))
    parquet_new = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet", "data_new.parquet"))

    df = pd.read_parquet(parquet_old)
    
    def get_print_years(sets):
        return sorted({set_year_dict.get(str(s).upper()) for s in sets if str(s).upper() in set_year_dict})
    # Apply the mapping
    df['print_years'] = df['sets'].apply(get_print_years)

    # Remove missed cards
    initial_count = len(df)
    for idx, row in df[df['print_years'].map(lambda x: len(x) == 0)].head(1).iterrows():
        print("Example of removed row:")
        print(row)
    df = df[df['print_years'].map(lambda x: len(x) > 0)]
    removed_count = initial_count - len(df)
    print(f"Rows removed due to missing printing year: {removed_count}\nRemaining rows: {len(df)}")

    df.to_parquet(parquet_new, index=False)
    
    df = pd.read_parquet(parquet_new)
    def extract_colors(mana_cost):
        if not isinstance(mana_cost, str):
            return []
        symbols = re.findall(r'{(.*?)}', mana_cost)
        symbols = [s for s in symbols if s in ['W','U','R','B','G']]
        return sorted(symbols)

    # Apply to DataFrame
    df['colors'] = df['manaCost'].apply(extract_colors)
    df.to_parquet(parquet_new)

    # Only keep monocolored
    df = df[df['colors'].apply(lambda x: isinstance(x, list) and len(x) == 1)]

    parquet_monocol = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "parquet", "monocol.parquet"))
    df.to_parquet(parquet_monocol)