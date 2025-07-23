import json
import requests
import pandas as pd
import os
import signal
import time
import sys
from contextlib import contextmanager

print("=== DEBUG: Script starting ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
try:
    print(f"Script location: {__file__}")
except NameError:
    print("Script location: <unknown - running in interactive mode>")

OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use

print(f"=== DEBUG: Configuration ===")
print(f"OLLAMA_NODE: {OLLAMA_NODE}")
print(f"BASE_URL: {BASE_URL}")
print(f"Model: {model}")

# Get timeout from environment variable (default 5 minutes = 300 seconds)
SUBJECT_TIME_LIMIT = int(os.environ.get('SUBJECT_TIME_LIMIT', 300))
print(f"SUBJECT_TIME_LIMIT: {SUBJECT_TIME_LIMIT} seconds")

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

print("=== DEBUG: Loading CSV files ===")

try:
    train_path = pd.read_csv("/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
    print(f"Train CSV loaded: {len(train_path)} rows")
except Exception as e:
    print(f"ERROR loading train CSV: {e}")
    sys.exit(1)

try:
    dev_path = pd.read_csv("/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")
    print(f"Dev CSV loaded: {len(dev_path)} rows")
except Exception as e:
    print(f"ERROR loading dev CSV: {e}")
    sys.exit(1)

id_train = train_path.iloc[:, 0].tolist()
id_dev = dev_path.iloc[:, 0].tolist()

print(f"Number of train subjects: {len(id_train)}")
print(f"Number of dev subjects: {len(id_dev)}")
print("First 3 train subjects:", id_train[:3] if len(id_train) >= 3 else id_train)
print("First 3 dev subjects:", id_dev[:3] if len(id_dev) >= 3 else id_dev)

all_subjects = [(subj, 'train') for subj in id_train] + [(subj, 'dev') for subj in id_dev]
print(f"Total subjects to process: {len(all_subjects)}")

# Test Ollama connection
print("=== DEBUG: Testing Ollama connection ===")
try:
    test_response = requests.get(f"http://{OLLAMA_NODE}:11434/api/tags", timeout=10)
    if test_response.status_code == 200:
        print("Ollama server is reachable")
        models = test_response.json().get('models', [])
        print(f"Available models: {[m.get('name') for m in models]}")
    else:
        print(f"Ollama server responded with status: {test_response.status_code}")
except Exception as e:
    print(f"Cannot reach Ollama server: {e}")
    print("This might be why the script is failing!")

results = []
processed_count = 0
skipped_count = 0
timeout_count = 0

# Output file path
output_file = "/home/users/nblair7/ai-psychiatrist/qual_resultsfin.csv"
print(f"Output file will be: {output_file}")

# Check if we should resume from existing results
if os.path.exists(output_file):
    print(f"Found existing results file: {output_file}")
    existing_results = pd.read_csv(output_file)
    completed_subjects = set(existing_results['participant_id'].tolist())
    print(f"Already completed {len(completed_subjects)} subjects")
    
    # Filter out already processed subjects
    all_subjects = [(subj, dtype) for subj, dtype in all_subjects if subj not in completed_subjects]
    print(f"Remaining subjects to process: {len(all_subjects)}")
    
    # Start with existing results
    results = existing_results.to_dict('records')
else:
    print("No existing results found, starting fresh")
    completed_subjects = set()

print(f"=== DEBUG: Starting main processing loop ===")
print(f"Will process {len(all_subjects)} subjects")

for i, (participant_id, dataset_type) in enumerate(all_subjects):
    print(f"\n=== DEBUG: Processing {i+1}/{len(all_subjects)}: {participant_id} ({dataset_type}) ===")
    
    start_time = time.time()
    
    try:
        with timeout_handler(SUBJECT_TIME_LIMIT):
            print(f"DEBUG: Looking for transcript files...")
            id_transcript = os.path.join("/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/", f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")
            
            print(f"DEBUG: Transcript path: {id_transcript}")
            
            if not os.path.exists(id_transcript):
                print(f"DEBUG: Transcript not found for {participant_id}")
                skipped_count += 1
                continue

            print(f"DEBUG: Transcript found, loading data...")
            
            transcript_data = pd.read_csv(id_transcript, sep='\t')
            print(f"DEBUG: Transcript loaded: {len(transcript_data)} rows")
            
            dialogue_txt = transcript_data.dropna(subset=['speaker', 'value'])
            print(f"DEBUG: After dropna: {len(dialogue_txt)} rows")
            
            dialogue_txt['dialogue'] = dialogue_txt['speaker'] + " : " + dialogue_txt['value']
            full_dialogue = "\n".join(dialogue_txt['dialogue'])
            
            print(f"DEBUG: Dialogue length: {len(full_dialogue)} characters")
            print(f"DEBUG: First 100 chars: {repr(full_dialogue[:100])}")
            
            # Your existing prompt setup
            system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible."
            qual_assessment = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
            {full_dialogue}
          Please:
          1. Provide an overall qualitative assessment of the subject's mental health. 
          2. Summarize social aspects that may influence the subject's mental health. (ex. familial relationships, frienship dynamics, work environment, etc. that are relevant to the subjects mental health)
          3. Summarize biological aspects that may influence the subject's mental health. (ex. famillial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc. that are relevant to the subjects mental health)
          4. Identify potential risk factors the subject may be experiencing.
          5. Use exact quotes from the transcript to support your assessment for each tag.

          Output should be formatted as bullet points with headings for each section using stars. Example: **Tiredness** <explanation of tiredness>. Do not include any additional text outside the bullet points
          Please answer in this XML format with each tag on a new line, properly indented. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:

          <assessment>
            <!-- Summary of participant's overall mental health -->
           <exact_quotes>
            <!-- Quotes from the transcript that support the assessment -->
            </exact_quotes>
          </assessment>

          <social_factors>
            <!-- Summary of social influences on patient's health -->
            <exact_quotes>
          </social_factors>

          <biological_factors>
            <!-- Summary of biological influences on patient's health -->
           <exact_quotes>
            <!-- Quotes from the transcript that support the assessment -->
            </exact_quotes>
          </biological_factors>

          <risk_factors>
            <!-- Summary of potential risk factors -->
             <exact_quotes>
           <!-- Quotes from the transcript that support the assessment -->
           </exact_quotes>
          </risk_factors>
        """
            
            print(f"DEBUG: Sending API request to {BASE_URL}")
            print(f"DEBUG: Request payload size: {len(json.dumps({'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': qual_assessment}]}))}")
            
            response = requests.post(
              BASE_URL,
              json = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": qual_assessment}],
                "stream": False,
                "options": {
                  "temperature": 0,
                  "top_k": 20,
                  "top_p": 0.9
                }
              },
              timeout=SUBJECT_TIME_LIMIT-10  # Leave 10 seconds buffer for processing
            )
            
            print(f"DEBUG: API response status: {response.status_code}")
            
            if response.status_code == 200:
                qual_content = response.json()['message']['content']
                print(f"DEBUG: API response received (length: {len(qual_content)} chars)")
                print(f"DEBUG: Response preview: {repr(qual_content[:200])}")
                
                result = {
                    "participant_id": participant_id,
                    "dataset_type": dataset_type,
                    "qualitative_assessment": qual_content
                }
                
                results.append(result)
                processed_count += 1
                elapsed_time = time.time() - start_time
                print(f"DEBUG: Completed participant {participant_id} in {elapsed_time:.1f}s ({processed_count} total completed)")
                
                # Save results incrementally every 10 subjects or immediately if first result
                if len(results) % 10 == 0 or len(results) == 1:
                    print(f"DEBUG: Saving results to {output_file}")
                    resultsdf = pd.DataFrame(results)
                    resultsdf.to_csv(output_file, index=False)
                    print(f"DEBUG: Saved progress: {len(results)} results to {output_file}")
                
            else:
                print(f"DEBUG: API request failed with status code: {response.status_code}")
                print(f"DEBUG: Response text: {response.text}")
                
    except TimeoutException as e:
        elapsed_time = time.time() - start_time
        print(f"DEBUG: TIMEOUT: {participant_id} exceeded {SUBJECT_TIME_LIMIT}s limit (ran for {elapsed_time:.1f}s)")
        timeout_count += 1
        
        # Save timeout result
        timeout_result = {
            "participant_id": participant_id,
            "dataset_type": dataset_type,
            "qualitative_assessment": f"TIMEOUT_ERROR: Processing exceeded {SUBJECT_TIME_LIMIT} seconds"
        }
        results.append(timeout_result)
        
        # Save progress after timeout
        print(f"DEBUG: Saving results after timeout to {output_file}")
        resultsdf = pd.DataFrame(results)
        resultsdf.to_csv(output_file, index=False)
        print(f"DEBUG: Saved progress after timeout: {len(results)} results")
        continue
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"DEBUG: ERROR processing {participant_id} after {elapsed_time:.1f}s: {str(e)}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print("DEBUG: Full traceback:")
        traceback.print_exc()
        continue

print(f"\n=== DEBUG: FINAL SUMMARY ===")
print(f"Total subjects: {len(all_subjects) + len(completed_subjects)}")
print(f"Previously completed: {len(completed_subjects)}")
print(f"Successfully processed this run: {processed_count}")
print(f"Skipped (no transcript): {skipped_count}")
print(f"Timed out: {timeout_count}")
print(f"Total results collected: {len(results)}")

if results:
    print(f"DEBUG: Final save to {output_file}")
    resultsdf = pd.DataFrame(results)
    resultsdf.to_csv(output_file, index=False)
    print(f"DEBUG: Final save completed: {output_file}")
else:
    print("DEBUG: No results to save!")

print("DEBUG: Script completed")