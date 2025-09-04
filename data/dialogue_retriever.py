import os
import pandas as pd

def save_transcript_texts(transcript_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(transcript_folder):
        if filename.endswith("_Transcript.csv"):
            file_path = os.path.join(transcript_folder, filename)
            try:
                df = pd.read_csv(file_path)
                lines = df["Text"].dropna().tolist()

                full_text = "\n".join(line.strip() for line in lines if line.strip())

                base_name = filename.replace("_Transcript.csv", "")
                out_file_path = os.path.join(output_folder, f"{base_name}.txt")

                with open(out_file_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                print(f"[✓] Saved {out_file_path}")

            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")

if __name__ == "__main__":
    transcript_folder = "./transcripts"
    output_folder = "./prompt_texts"

    print(f"Processing transcripts from: {transcript_folder}")
    save_transcript_texts(transcript_folder, output_folder)
    print("All transcripts processed.")
