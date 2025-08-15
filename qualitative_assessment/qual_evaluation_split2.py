import pandas as pd
import requests
import time
import json
import re
import os
from requests.exceptions import Timeout, RequestException

# only include failed IDs
df = df[df['participant_id'].isin(failed_participant_ids)]
print(f"Filtered to {len(df)} failed participants to process")

if len(df) == 0:
    print("ERROR: No matching participants found in the CSV file!")
    print("Make sure the participant_id column exists and contains the expected values")
    exit()


results = []
failed_evaluations = []
processed_count = 0


def parse_score_and_explanation(response_text):
    """
    Extract score and explanation from model response

    Parameters
    ----------
   response_text : str
   Output from the model containing scores and explanation.

    Returns
    -------
    int or None, str
        score :   int or None
            The extracted score between 1 and 10, if found. Returns None if no valid score is found.
        explanation : str 
              The original response text without leading/trailing whitespace.       
    """
    score_patterns = [
        r'score[:\s]*(\d+)',
        r'(\d+)[/\s]*(?:out of\s*)?10',
        r'(\d+)[/\s]*10',
        r'rating[:\s]*(\d+)',
        r'^(\d+)',  # Number at start of line
    ]
    
    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_score = int(match.group(1))
            if 1 <= potential_score <= 10:
                score = potential_score
                break
    
    return score, response_text.strip()

# Configuration
OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use

# File paths
input_csv_path = "/data/users2/nblair7/analysis_results/qual_resultsfin.csv"
output_csv_path = "/data/users2/nblair7/analysis_results/eval_results_split.csv"  

# failed IDs
failed_participant_ids = [
    478, 479, 485, 486, 487, 488, 491, 302, 307, 331, 335, 346, 367, 377, 381, 382, 
    388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 451, 
    458, 472, 476, 477, 482, 483, 484, 489, 490, 492
]

print(f"Processing only these {len(failed_participant_ids)} failed participants: {failed_participant_ids}")

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv(input_csv_path)
print(f"Loaded {len(df)} total participants")


if os.path.exists(output_csv_path):
    print(f"Found existing results file: {output_csv_path}")
    existing_results = pd.read_csv(output_csv_path)
    
    
    good_results = existing_results[~existing_results['participant_id'].isin(failed_participant_ids)]
    print(f"Preserving {len(good_results)} existing good results")
    
    completed_failed = existing_results[existing_results['participant_id'].isin(failed_participant_ids)]
    completed_subjects = set(completed_failed['participant_id'].tolist())
    print(f"Already completed {len(completed_subjects)} of the failed subjects")
    
    df = df[~df['participant_id'].isin(completed_subjects)]
    print(f"Remaining failed subjects to process: {len(df)}")
    results = existing_results.to_dict('records')
else:
    print("No existing results found, starting fresh")
    completed_subjects = set()


for index, row in df.iterrows():
    participant_id = row['participant_id']
    qualitative_assessment = row['qualitative_assessment']
    
    print(f"\n--- Processing {index + 1}/{len(df)}: {participant_id} (FAILED RETRY) ---")
    
    start_time = time.time()
    
    # metric prompts
    coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only.

Score from 1-10 where higher scores indicate better performance:
- Coherence (1–10): Is the response logically consistent?

Format your response as:
Score: [your score]


---
{qualitative_assessment}
---"""

    completeness_prompt = f"""Evaluate the following qualitative assessment output for COMPLETENESS only.

Score from 1-10 where higher scores indicate better performance:
- Completeness (1–10): Does the assessment cover all relevant symptoms, severities, duration/frequency?

Format your response as:
Score: [your score]


---
{qualitative_assessment}
---"""

    specificity_prompt = f"""Evaluate the following qualitative assessment output for SPECIFICITY only.

Score from 1-10 where higher scores indicate better performance:
- Specificity (1–10): Does it avoid vague/generic statements like 'the patient seems depressed'?

Format your response as:
Score: [your score]


---
{qualitative_assessment}
---"""

    accuracy_prompt = f"""Evaluate the following qualitative assessment output for ACCURACY only.

Score from 1-10 where higher scores indicate better performance:
- Accuracy (1–10): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Are there factual inconsistencies?

Format your response as:
Score: [your score]


---
{qualitative_assessment}
---"""

    collective_prompt = f"""Provide a brief collective explanation/summary of the overall quality of this qualitative assessment, considering all aspects: coherence, completeness, specificity, and accuracy together. Only use 2-3 sentences.

---
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
        "messages": [{"role": "user", "content": completeness_prompt} ],
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
    
    collective_request = {
        "model": model,
        "messages": [{"role": "user", "content": collective_prompt}],
        "stream": False,
        "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
    }
    
    timeout = 300  # 5 minutes
    
    try:
        
        result = {'participant_id': participant_id}
        
        # coherence
        print("  Getting coherence response...")
        coherence_response = requests.post(BASE_URL, json=coherence_request, timeout=timeout-10)
        if coherence_response.status_code == 200:
            coherence_content = coherence_response.json()['message']['content']
            coherence_score, coherence_explanation = parse_score_and_explanation(coherence_content)
            result['coherence'] = coherence_score
            print(f"  Coherence score: {coherence_score}")
        else:
            result['coherence'] = None
        
        time.sleep(2)
        
        # completeness
        print("  Getting completeness response...")
        completeness_response = requests.post(BASE_URL, json=completeness_request, timeout=timeout-10)
        if completeness_response.status_code == 200:
            completeness_content = completeness_response.json()['message']['content']
            completeness_score, completeness_explanation = parse_score_and_explanation(completeness_content)
            result['completeness'] = completeness_score
            print(f"  Completeness score: {completeness_score}")
        else:
            result['completeness'] = None
        
        time.sleep(2)
        
        # specificity
        print("  Getting specificity response...")
        specificity_response = requests.post(BASE_URL, json=specificity_request, timeout=timeout-10)
        if specificity_response.status_code == 200:
            specificity_content = specificity_response.json()['message']['content']
            specificity_score, specificity_explanation = parse_score_and_explanation(specificity_content)
            result['specificity'] = specificity_score
            print(f"  Specificity score: {specificity_score}")
        else:
            result['specificity'] = None
        
        time.sleep(2)
        
        # accuaracy
        print("  Getting accuracy response...")
        accuracy_response = requests.post(BASE_URL, json=accuracy_request, timeout=timeout-10)
        if accuracy_response.status_code == 200:
            accuracy_content = accuracy_response.json()['message']['content']
            accuracy_score, accuracy_explanation = parse_score_and_explanation(accuracy_content)
            result['accuracy'] = accuracy_score
            print(f"  Accuracy score: {accuracy_score}")
        else:
            result['accuracy'] = None
        
        time.sleep(2)
        
        # collective explanation
        print("  Getting collective explanation...")
        collective_response = requests.post(BASE_URL, json=collective_request, timeout=timeout-10)
        if collective_response.status_code == 200:
            result['collective_explanation'] = collective_response.json()['message']['content']
            print(f"  Collective: {result['collective_explanation'][:50]}...")
        else:
            result['collective_explanation'] = 'API request failed'
        
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
            'collective_explanation': f"Error: {e}",
        }
        results.append(result)
    
    # Save every 10 ids
    if len(results) % 10 == 0 or len(results) == 1:
        resultsdf = pd.DataFrame(results)
        resultsdf.to_csv(output_csv_path, index=False)
        print(f"Saved progress: {len(results)} results to {output_csv_path}")
    
    
    time.sleep(1)

# Summary
print(f"summary of processing:")
print(f"Total failed subjects to retry: {len(failed_participant_ids)}")
print(f"Previously completed in retry file: {len(completed_subjects)}")
print(f"Successfully processed this run: {processed_count}")
print(f"Total results collected: {len(results)}")

# Final results
if results:
    resultsdf = pd.DataFrame(results)
    resultsdf.to_csv(output_csv_path, index=False)
    print(f"Final save completed: {output_csv_path}")
    print(f"\nCSV columns created:")
    print(f"- participant_id")
    print(f"- coherence")
    print(f"- completeness") 
    print(f"- specificity")
    print(f"- accuracy")
    print(f"- collective_explanation")
else:
    print("No results to save!")

print(f"\nFailed participant IDs processed: {failed_participant_ids}")
