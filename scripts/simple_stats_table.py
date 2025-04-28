import json
from json import JSONDecodeError
from typing import Optional, Dict, List
import pandas as pd
import markdown

def extract_messages(conv: Dict) -> Optional[pd.DataFrame]:
    """Extracts messages from a single conversation dictionary."""
    if not isinstance(conv, dict) or "conversations" not in conv:
        return None

    messages = []
    for turn in conv["conversations"]:
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            print(f"Warning: Skipping invalid turn: {turn}")
            continue

        messages.append({
            "from": turn["from"],
            "value": turn["value"],
        })

    if messages:
        return pd.DataFrame(messages)
    return None


def process_jsonl_file(file_path: str) -> Optional[pd.DataFrame]:
    """Reads a JSONL file, parses, extracts messages, returns a DataFrame."""
    all_messages = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                except JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} - Line: {line}")
                    continue
                if "conversations" not in conv:
                    print(f"Warning: Skipping line without 'conversations': {line}")
                    continue
                messages_df = extract_messages(conv)
                if messages_df is not None:
                    all_messages.append(messages_df)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    if all_messages:
        return pd.concat(all_messages, ignore_index=True)
    else:
        print("No valid messages found in the file.")
        return None


def calculate_metrics(df: pd.DataFrame, convs: List[Dict]) -> Dict:
    """Calculates metrics for the table."""

    metrics = {}
    metrics["No. of dialogues"] = len(convs)
    metrics["Total no. of turns"] = len(df)
    metrics["Avg. turns per dialogue"] = len(df) / len(convs) if len(convs) > 0 else 0
    all_tokens = []
    for text in df["value"]:
        all_tokens.extend(text.split())
    metrics["Avg. tokens per turn"] = len(all_tokens) / len(df) if len(df) > 0 else 0
    metrics["Total unique tokens"] = len(set(all_tokens))
    metrics["No. of domains"] = 0  # Placeholder.
    metrics["No. of slots"] = 0    # Placeholder.
    metrics["No. of slot values"] = 0 # Placeholder.
    return metrics


def get_all_conversations(file_path: str) -> List[Dict]:
    """Reads jsonl file and returns list of conversation dictionaries."""
    conversations = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                    if "conversations" in conv:
                        conversations.append(conv)
                except JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} - Line: {line}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return conversations


def save_markdown_as_styled_html(markdown_text: str, output_file: str):
    """Converts markdown to HTML, adds CSS styling, and saves to file."""
    
    #Crucially, tell markdown to render tables correctly.
    html = markdown.markdown(markdown_text, extensions=['tables'])

    # Add CSS styling for a more visually appealing table
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Dataset Metrics</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif; /* Use Arial, a common sans-serif font */
            margin: 40px;          /* Increase margin for better spacing */
            background-color: #f4f4f4; /* Light gray background for the page */
        }}
        table {{
            border-collapse: collapse;
            width: 70%;              /* Wider table */
            margin: 30px auto;       /* Center the table and add more vertical margin */
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); /* Stronger shadow */
            border-radius: 8px;      /* Rounded corners for the table */
            overflow: hidden;      /* Ensure rounded corners work correctly */
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 15px;            /* More padding */
            text-align: left;
        }}
        th {{
            background-color: #3498db;   /* Blue header background */
            color: white;               /* White text for the header */
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #ecf0f1;    /* Lighter alternate row color */
        }}
        tr:hover {{
            background-color: #d4e6f1;  /* Hover effect */
        }}
    </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(styled_html)
    except Exception as e:
        print(f"An unexpected error occurred while writing to file: {e}")


def main():
    file_path = "/Users/dre/Downloads/atk_generated_datasets/crossfire/all.jsonl"
    output_html_file = "/Users/dre/Downloads/crossfire_data_md.html"  # Name of the output HTML file
    all_messages_df = process_jsonl_file(file_path)
    all_convs = get_all_conversations(file_path)

    if all_messages_df is not None:
        metrics = calculate_metrics(all_messages_df, all_convs)

        # Create the table
        table_data = {
            "Metric ↓ Dataset →": [
                #"No. of domains",
                "No. of dialogues",
                "Total no. of turns",
                "Avg. turns per dialogue",
                "Avg. tokens per turn",
                "Total unique tokens",
                #"No. of slots",
                #"No. of slot values",
            ],
            "My Dataset": [
                #metrics["No. of domains"],
                metrics["No. of dialogues"],
                metrics["Total no. of turns"],
                f"{metrics['Avg. turns per dialogue']:.2f}",
                f"{metrics['Avg. tokens per turn']:.2f}",
                metrics["Total unique tokens"],
                #metrics["No. of slots"],
                #metrics["No. of slot values"],
            ],
        }
        table_df = pd.DataFrame(table_data)
        markdown_table = table_df.to_markdown(index=False)
        print(markdown_table)

        save_markdown_as_styled_html(markdown_table, output_html_file)
        print(f"Markdown table saved as styled HTML to {output_html_file}")

if __name__ == "__main__":
    main()
