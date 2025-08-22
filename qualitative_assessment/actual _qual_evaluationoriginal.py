import os
import pandas as pd
import requests
import time
import json
import re
from requests.exceptions import Timeout, RequestException

def parse_score_and_explanation(response_text):
    """Extract score and explanation from model response"""
    score_patterns = [
        r'score[:\s]*(\d+)',
        r'(\d+)[/\s]*(?:out of\s*)?5',
        r'(\d+)[/\s]*5',
        r'rating[:\s]*(\d+)',
        r'^(\d+)',  # Number at start of line
    ]
    
    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_score = int(match.group(1))
            if 1 <= potential_score <= 5:
                score = potential_score
                break
    
    return score, response_text.strip()

# Configuration
OLLAMA_NODE = "arctrddgxa002" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use

# File paths
input_csv_path = "/data/users2/nblair7/analysis_results/qual_resultsfin.csv"
output_csv_path = "/data/users2/nblair7/analysis_results/eval_results_new.csv" 

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv(input_csv_path)
print(f"Loaded {len(df)} participants")

results = []
failed_evaluations = []
processed_count = 0
skipped_count = 0

import os
if os.path.exists(output_csv_path):
    print(f"Found existing results file: {output_csv_path}")
    existing_results = pd.read_csv(output_csv_path)
    completed_subjects = set(existing_results['participant_id'].tolist())
    print(f"Already completed {len(completed_subjects)} subjects")
    
    df = df[~df['participant_id'].isin(completed_subjects)]
    print(f"Remaining subjects to process: {len(df)}")
    
    results = existing_results.to_dict('records')
else:
    print("No existing results found, starting fresh")
    completed_subjects = set()

for index, row in df.iterrows():
    participant_id = row['participant_id']
    qualitative_assessment = row['qualitative_assessment']
    
    print(f"\n--- Processing {index + 1}/{len(df)}: {participant_id} ---")
    
    # Load transcript for this participant (replacing the old function)
    id_transcript = os.path.join("/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/", f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")
    print(f"Looking for transcript at: {id_transcript}")
    
    if not os.path.exists(id_transcript):
        print(f"Transcript not found for {participant_id}")
        skipped_count += 1
        continue
    
    print(f"Transcript found, loading data...")
    
    try:
        transcript_data = pd.read_csv(id_transcript, sep='\t')
        dialogue_txt = transcript_data.dropna(subset=['speaker', 'value'])
        dialogue_txt['dialogue'] = dialogue_txt['speaker'] + " : " + dialogue_txt['value']
        transcript = "\n".join(dialogue_txt['dialogue'])
        
        print(f"Dialogue length: {len(transcript)} characters")
        print(f"First 200 chars: {transcript[:200]}...")
        
    except Exception as e:
        print(f"Error loading transcript for {participant_id}: {e}")
        skipped_count += 1
        continue
    
    start_time = time.time()
    
    #scoring criteria
    #score = "You are an expert psychiatrist evaluating qualitative assessments of mental health conditions. Evaluate on a score of 1-5 for the metric based on the following criteria: A score of 5 is 0 mistakes. A score of 4 is 1-2 mistakes. A score of 3 is 3-4 mistakes. A score of 2 is 5-6 mistakes. A score of 1 is 7 or more mistakes"
    
    # metric prompts
    coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only. Compare qualitative_assessment to the provided transcript. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.
Score from 1-5 solely based on the following criteria: 
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes

Coherence (1–5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions within the assessment.

Format your response as:
Score: [your score]
Explanation: [explanation]

---
Here is the transcript: 
{transcript}

Here is the assessment based on the transcript: 
{qualitative_assessment}
---"""

    completeness_prompt = f"""Evaluate the following qualitative assessment output for COMPLETENESS only. Compare qualitative_assessment to the provided transcript. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

Score from 1-5 solely based on the following criteria: 
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes

Completeness (1–5): Does the assessment cover all relevant symptoms, severities, duration/frequency? Mistakes have missed PHQ-8 symptoms, or duration/frequency details.

Format your response as:
Score: [your score]
Explanation: [explanation]

Here is the transcript: 
{transcript}

Here is the assessment based on the transcript: 
{qualitative_assessment}
---"""

    specificity_prompt = f"""Evaluate the following qualitative assessment output for SPECIFICITY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

Score from 1-5 solely based on the following criteria: 
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes

specificity (1–5): Is the assessment specific? Mistakes include using vague/generic statements like 'the patient seems depressed'.

Format your response as:
Score: [your score]
Explanation: [explanation]


---
Here is the transcript: 
{transcript}

Here is the assessment based on the transcript: 
{qualitative_assessment}
---"""

    accuracy_prompt = f"""Evaluate the following qualitative assessment output for ACCURACY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

Score from 1-5 solely based on the following criteria: 
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes

Accuracy (1–5): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Mistakes are incorrect symptoms or incorrect duration/frequecy. 

Format your response as:
Score: [your score]
Explanation: [explanation]


---
Here is the transcript: 
{transcript}

Here is the assessment based on the transcript: 
{qualitative_assessment}
---"""

    # requests for each metric
    coherence_request = {
        "model": model,
        "messages": [{"role": "user", "content": coherence_prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
    }
    
    completeness_request = {
        "model": model,
        "messages": [{"role": "user", "content": completeness_prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
    }
    
    specificity_request = {
        "model": model,
        "messages": [{"role": "user", "content": specificity_prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
    }
    
    accuracy_request = {
        "model": model,
        "messages": [{"role": "user", "content": accuracy_prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
    }
    
    
    timeout = 300  
    
    try:
        result = {'participant_id': participant_id}
        
        # coherence
        print("  Getting coherence response...")
        coherence_response = requests.post(BASE_URL, json=coherence_request, timeout=timeout-10)
        if coherence_response.status_code == 200:
            coherence_content = coherence_response.json()['message']['content']
            coherence_score, _ = parse_score_and_explanation(coherence_content)
            result['coherence'] = coherence_score
            result['coherence_explanation'] = coherence_content  # Store full response
            print(f"  Coherence score: {coherence_score}")
        else:
            result['coherence'] = None
            result['coherence_explanation'] = f"API Error: {coherence_response.status_code}"
        
        time.sleep(2)
        
        # completeness
        print("  Getting completeness response...")
        completeness_response = requests.post(BASE_URL, json=completeness_request, timeout=timeout-10)
        if completeness_response.status_code == 200:
            completeness_content = completeness_response.json()['message']['content']
            completeness_score, _ = parse_score_and_explanation(completeness_content)
            result['completeness'] = completeness_score
            result['completeness_explanation'] = completeness_content  # Store full response
            print(f"  Completeness score: {completeness_score}")
        else:
            result['completeness'] = None
            result['completeness_explanation'] = f"API Error: {completeness_response.status_code}"
        
        time.sleep(2)
        
        # specificity
        print("  Getting specificity response...")
        specificity_response = requests.post(BASE_URL, json=specificity_request, timeout=timeout-10)
        if specificity_response.status_code == 200:
            specificity_content = specificity_response.json()['message']['content']
            specificity_score, _ = parse_score_and_explanation(specificity_content)
            result['specificity'] = specificity_score
            result['specificity_explanation'] = specificity_content  # Store full response
            print(f"  Specificity score: {specificity_score}")
        else:
            result['specificity'] = None
            result['specificity_explanation'] = f"API Error: {specificity_response.status_code}"
        
        time.sleep(2)
        
        # accuaracy
        print("  Getting accuracy response...")
        accuracy_response = requests.post(BASE_URL, json=accuracy_request, timeout=timeout-10)
        if accuracy_response.status_code == 200:
            accuracy_content = accuracy_response.json()['message']['content']
            accuracy_score, _ = parse_score_and_explanation(accuracy_content)
            result['accuracy'] = accuracy_score
            result['accuracy_explanation'] = accuracy_content  # Store full response
            print(f"  Accuracy score: {accuracy_score}")
        else:
            result['accuracy'] = None
            result['accuracy_explanation'] = f"API Error: {accuracy_response.status_code}"
        
        time.sleep(2)
        

        
        results.append(result)
        processed_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"Completed participant {participant_id} in {elapsed_time:.1f}s ({processed_count} total completed)")
            
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        result = {
            'participant_id': participant_id,
            'coherence': None,
            'completeness': None,
            'specificity': None,
            'accuracy': None,
            'coherence_explanation': f"Error: {e}",
            'completeness_explanation': f"Error: {e}",
            'specificity_explanation': f"Error: {e}",
            'accuracy_explanation': f"Error: {e}",
        }
        results.append(result)
        failed_evaluations.append(participant_id)
    
    # Save progress every 10 participants
    if len(results) % 10 == 0 or len(results) == 1:
        resultsdf = pd.DataFrame(results)
        resultsdf.to_csv(output_csv_path, index=False)
        print(f"Saved progress: {len(results)} results to {output_csv_path}")
    
    time.sleep(1)

# Final summary
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total subjects in input: {len(df) + len(completed_subjects)}")
print(f"Previously completed: {len(completed_subjects)}")
print(f"Attempted this run: {len(df)}")
print(f"Skipped (no transcript): {skipped_count}")
print(f"Successfully processed: {processed_count}")
print(f"Failed: {len(failed_evaluations)}")
print(f"Total results collected: {len(results)}")

if failed_evaluations:
    print(f"Failed participant IDs: {failed_evaluations}")

# Save final results
if results:
    resultsdf = pd.DataFrame(results)
    resultsdf.to_csv(output_csv_path, index=False)
    print(f"Final save completed: {output_csv_path}")
    print(f"\nCSV columns created:")
    print(f"- participant_id")
    print(f"- coherence / coherence_explanation")
    print(f"- completeness / completeness_explanation") 
    print(f"- specificity / specificity_explanation")
    print(f"- accuracy / accuracy_explanation")
else:
    print("No results to save!")