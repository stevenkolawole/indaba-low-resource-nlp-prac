import subprocess
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_models():
    config = PeftConfig.from_pretrained("alizaidi/lora-mt5-goud")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = PeftModel.from_pretrained(
        base_model, "alizaidi/lora-mt5-goud", device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained("alizaidi/lora-mt5-goud")

    return model, tokenizer, config


def install_requirements(requirements_path="requirements.txt"):
    process = subprocess.Popen(
        ["pip", "install", "-r", requirements_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Ensure the output is captured as text, not bytes
    )

    # Capture and print the output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    # Check for any errors
    if process.returncode != 0:
        print("\nError during installation:")
        for line in process.stderr:
            print(line, end="")


def download_and_extract_zip(url, extract_to="."):
    import requests, zipfile, io

    print(f"Starting download from {url}...")

    # Start the download process
    response = requests.get(url, stream=True)

    # Check if the download was successful
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        print(f"Download complete. Extracting {total_size / (1024 * 1024):.2f} MB...")

        # Create a ZipFile object from the downloaded content
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all the contents of the zip file into the specified directory
            z.extractall(path=extract_to)

            # Print the names of the extracted files
            extracted_files = z.namelist()
            print(f"Extracted {len(extracted_files)} files to '{extract_to}':")
            for file in extracted_files:
                print(f" - {file}")

        print("Download and extraction complete.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
