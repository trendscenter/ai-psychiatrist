import json
import requests
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Union
import csv
from datetime import datetime
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import sys

############################################################################################################################
##################### Configuring ollama, paths, and pydantic model s######################################################
############################################################################################################################


# Ollama Config
OLLAMA_NODE = "arctrdagn039"
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3-optimized:27b"

# Output paths
OUTPUT_DIR = r"/data/users2/agreene46/ai-psychiatrist/analysis_output"
JSONL_FILE = os.path.join(OUTPUT_DIR, "oss_embedding_results_analysis.jsonl")
LOG_FILE = os.path.join(OUTPUT_DIR, "oss_embedding_log_file.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PHQ8ScoreWithExplanation(BaseModel):
    evidence: str  # Direct quotes or references from interview
    reason: str  # Brief explanation for the score
    score: Union[int, str]  # 0-3 or "N/A"

class PHQ8ScoresWithExplanations(BaseModel):
    PHQ8_NoInterest: PHQ8ScoreWithExplanation  # Little interest or pleasure in doing things
    PHQ8_Depressed: PHQ8ScoreWithExplanation   # Feeling down, depressed, or hopeless
    PHQ8_Sleep: PHQ8ScoreWithExplanation       # Trouble falling/staying asleep, or sleeping too much
    PHQ8_Tired: PHQ8ScoreWithExplanation       # Feeling tired or having little energy
    PHQ8_Appetite: PHQ8ScoreWithExplanation    # Poor appetite or overeating
    PHQ8_Failure: PHQ8ScoreWithExplanation     # Feeling bad about self or like a failure
    PHQ8_Concentrating: PHQ8ScoreWithExplanation # Trouble concentrating
    PHQ8_Moving: PHQ8ScoreWithExplanation      # Moving/speaking slowly or being fidgety/restless

############################################################################################################################
##################### Created 4 line chunks (with a 2 lines sliding window) for each reference transcript and embedded them 
############################################################################################################################

def get_embedding(text, model="dengcao/Qwen3-Embedding-8B:Q8_0"):
    """
    Creates embedding from given text input and model 

    Parameters
    ----------
    text : string
        The text to be embedded
    model : string
        The name of the ollama model to be used for embedding

    Returns
    -------
    list
        The vector embedding of the text
    """
    BASE_URL = f"http://{OLLAMA_NODE}:11434/api/embeddings"
    
    response = requests.post(
        BASE_URL,
        json={
            "model": model,
            "prompt": text
        }
    )
    
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")

def create_sliding_chunks(transcript_text, chunk_size=4, step_size=2):
    """
    Splits the transcript into several chunks 

    Parameters
    ----------
    transcript_text : string
        The transcript
    chunk_size : int
        The amount of newlines per chunk
    step_size : int
        The newline distance moved each time a chunk is created
            -Ex. transcript_text = "A\nB\nC\nD\nE\nF\nG\nH", chunk_size = 4, step_size = 2
            -Chunk 1: "A\nB\nC\nD"
            -Chunk 2: "C\nD\nE\nF"

    Returns
    -------
    list
        The text chunk strings
    """
    lines = transcript_text.split('\n')
    
    # Remove empty lines at the end if any
    while lines and lines[-1] == '':
        lines.pop()
    
    chunks = []
    
    # If fewer lines than chunk_size, just return the whole thing
    if len(lines) <= chunk_size:
        return ['\n'.join(lines)]
    
    # Create sliding windows
    for i in range(0, len(lines) - chunk_size + 1, step_size):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)
    
    # If the last chunk doesn't include the final lines, add one more chunk
    last_chunk_start = len(lines) - chunk_size
    if last_chunk_start > 0 and (last_chunk_start % step_size) != 0:
        final_chunk = '\n'.join(lines[last_chunk_start:])
        if final_chunk not in chunks:
            chunks.append(final_chunk)
    
    return chunks

############################################################################################################################
######### Runs cosine similarity to pull 3 most similar transcript chunks from the reference transcripts ##################
############################################################################################################################

def find_similar_chunks(evidence_text_embedding, participant_embedded_transcripts, top_k=3):
    """
    Runs cosine similarity between the evidence and all of the reference transcript embedded chunks.
    Then, grabs the top_k most similar ones.

    Parameters
    ----------
    evidence_text_embedding : list
        The embedding of the pulled evidence from the current transcript for the given PHQ8 question
    participant_embedded_transcripts : dict
        The dictionary with participant IDs as keys and (raw_text, embedding) as values for each transcript chunk
    top_k : int
        The number of most similar chunks that should be pulled (ex. top_k=3 means pull the 3 most similar chunks)

    Returns
    -------
    list
        List of dictionaries containing the most similar chunks, each with keys:
        'participant_id', 'raw_text', 'similarity', and 'embedding'.
        Sorted by similarity score in descending order.
    """
    similarities = []
    
    # Go through all participants and their embeddings
    for participant_id, embeddings_array in participant_embedded_transcripts.items():
        for i, (raw_text, embedding) in enumerate(embeddings_array):
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [evidence_text_embedding], 
                [embedding]
            )[0][0]
            
            similarities.append({
                'participant_id': participant_id,
                'raw_text': raw_text,
                'similarity': similarity,
                'embedding': embedding
            })
    
    # Sort by similarity and get top 3 results
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

def log_message(message, print_to_console=True):
    """
    Logs a given message in the log file

    Parameters
    ----------
    message : string
        The message to be embedded in the log file

    print_to_console : bool
        Whether or not to print the given message to the console

    Writes
    -------
    string
        Writes the given message to the log file and prints to terminal if the bool is true
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')
    
    if print_to_console:
        print(message)

def load_processed_participants():
    """
    Loads all participants ids from the jsonl file so as to skip them when processing

    Returns
    -------
    set
        All processed participant id ints
    """
    processed_ids = set()
    
    if os.path.exists(JSONL_FILE):
        try:
            with open(JSONL_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed_ids.add(data['participant_id'])
                    except:
                        continue
        except Exception as e:
            log_message(f"Could not read existing JSONL file: {str(e)}")
    
    return processed_ids

def save_participant_result(participant_id, phq8_scores):
    """
    Saves a given participants prediction output to the jsonl file

    Parameters
    ----------
    participant_id : int
        The id of the given participant

    phq8_scores : PHQ8ScoresWithExplanations
        Pydantic model containing PHQ-8 assessment results. Has 8 attributes 
        (PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired, PHQ8_Appetite, 
        PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving), each of type 
        PHQ8ScoreWithExplanation with 'evidence', 'reason', and 'score' fields.
    
    Writes
    -------
    json
        Record containing participant_id and timestamp. 
        Also, predicted participants PHQ8 information for each question including:
        "evidence", "reason", and "score"
    """
    result = {
        "participant_id": participant_id,
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract scores and explanations
    for key in ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 
                'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']:
        score_data = getattr(phq8_scores, key)
        result[key] = {
            "evidence": score_data.evidence,
            "reason": score_data.reason,
            "score": score_data.score
        }
    
    # Append to JSONL file
    with open(JSONL_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

############################################################################################################################
####### Extracting evidence from the transcript for each PHQ8 question using Gemma 3 27b ###################################
############################################################################################################################

def extract_evidence_for_participant(transcript, participant_id):
    """
    Has LLM grab evidence from the transcript based on the PHQ8 criteria.

    Parameters
    ----------
    transcript : string
        The participants transcript

    participant_id : int
        The id of the given participant
    
    Returns
    -------
    dict
        A dictionary containing PHQ-8 domain keys mapped to lists of evidence quotes.
        Each key corresponds to a PHQ-8 domain (e.g., 'PHQ8_NoInterest', 'PHQ8_Depressed')
        and each value is a list of direct transcript quotes that provide evidence
        for that domain. Empty lists indicate no evidence found for that domain.
        Duplicate quotes are removed from each evidence list.
    """
    evidence_extraction_prompt = f"""Analyze the following therapy transcript and extract specific text chunks that provide evidence for each PHQ-8 domain. Based on the evidence found, determine the appropriate PHQ-8 score for each domain.

PHQ-8 scoring reference:
- 0 = not at all (0-1 days)
- 1 = several days (2-6 days) 
- 2 = more than half the days (7-11 days)
- 3 = nearly every day (12-14 days)

For each domain below, extract direct quotes from the transcript that indicate the presence, absence, frequency, or severity of symptoms. Then determine the most appropriate score based on the evidence.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain. Each quote should be a direct excerpt from the transcript that supports your scoring decision. Include as many relevant quotes as you find - this could range from 0 quotes (if no evidence) to multiple quotes per domain.

Therapy transcript:
{transcript}

Respond with valid JSON matching this structure:
{{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2", "evidence_3", "evidence_4"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2", "evidence_3"],
    "PHQ8_Tired": ["evidence_1", "evidence_2"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1", "evidence_2", "evidence_3", "evidence_4", "evidence_5"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1", "evidence_2"]
}}

Important: Extract UNIQUE quotes only - do not repeat the same quote multiple times. Each quote should be different and provide distinct, related evidence. If no evidence exists for a domain, return an empty array for that domain. Also, do not format the evidence grabbed in any way, output it EXACTLY as it is in the transcript.
"""

    try:
        response = requests.post(
            BASE_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": evidence_extraction_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_k": 10,
                    "top_p": 0.8
                }
            }
        )
        
        response_data = response.json()
        content = response_data['message']['content']
        content = content.strip('```json\n').strip('\n```')
        
        evidence_dict = json.loads(content)
        
        # Remove duplicate quotes in each evidence list
        for key in evidence_dict:
            if isinstance(evidence_dict[key], list):
                evidence_dict[key] = list(dict.fromkeys(evidence_dict[key]))
        
        return evidence_dict
        
    except Exception as e:
        log_message(f"Error extracting evidence for participant {participant_id}: {str(e)}", print_to_console=False)
        raise

############################################################################################################################
######## Grabbing similar chunks + their PHQ8 score from reference transcripts based on current transcript evidence.
#############################################################################################################################

def process_evidence_for_references(evidence_dict, participant_embedded_transcripts, phq8_ground_truths):
    """
    Grabs chunks from other transcripts that are similar to the evidence pulled for the current transcript.
    Then, grabs those chunks ground truth scores and formats all that into a string for use in the prompt.

    Parameters
    ----------
    evidence_dict : dict
        Dictionary containing PHQ-8 domain keys mapped to lists of evidence quotes
    
    participant_embedded_transcripts : dict
        The dictionary with participant IDs as keys and (raw_text, embedding) as values for each transcript chunk
    
    phq8_ground_truths : pandas dataframe
        Dataframe containing ground truth PHQ-8 scores for participants
    
    Returns
    -------
    str
        A string with all the reference transcripts chunks and their corresponding PHQ8 scores
    """

    evidence_keys = [
        'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
        'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
    ]
    
    all_references = []
    
    for evidence_key in evidence_keys:
        # Get evidence texts for this key
        evidence_texts = evidence_dict.get(evidence_key, [])
        
        # Skip if empty
        if not evidence_texts:
            continue
        
        # Combine evidence texts into single string
        combined_text = '\n'.join(evidence_texts)
        
        # Skip if less than 15 characters
        if len(combined_text) < 15:
            continue
        
        print(f"Processing {evidence_key}...")
        
        try:
            # Get embedding for combined evidence text
            evidence_embedding = get_embedding(combined_text)
            
            # Find top 3 similar chunks
            similar_chunks = find_similar_chunks(
                evidence_embedding, 
                participant_embedded_transcripts, 
                top_k=3
            )
            
            # Add each reference with its own header
            for chunk_info in similar_chunks:
                participant_id = chunk_info['participant_id']
                raw_text = chunk_info['raw_text']
                
                # Get the ground truth score for this participant and evidence type
                participant_data = phq8_ground_truths.loc[
                    phq8_ground_truths['Participant_ID'] == participant_id
                ]
                
                if not participant_data.empty:
                    score = int(participant_data[evidence_key].values[0])
                    reference_entry = f"({evidence_key} Score: {score})\n{raw_text}"
                    all_references.append(reference_entry)
                else:
                    print(f"No ground truth data found for participant {participant_id}")
                
        except Exception as e:
            print(f"Error processing {evidence_key}: {e}")
            continue
    
    # Combine all references into a string
    if all_references:
        reference_evidence = "<Reference Examples>\n\n" + "\n\n".join(all_references) + "\n\n<Reference Examples>"
    else:
        reference_evidence = "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
    
    return reference_evidence

################################################################################################################################
############ Running quantitative analysis, but with reference transcripts this time. Grabs evidence, reasons, and outputs score
#################################################################################################################################


def run_phq8_analysis_batch(patient_transcript, ground_truth, reference_evidence=""):
    """
    Grabs chunks from other transcripts that are similar to the evidence pulled for the current transcript.
    Then, grabs those chunks ground truth scores and formats all that into a string for use in the prompt.

    Parameters
    ----------
    patient_transcript : string
        The patients transcript
    
    ground_truth : dict
        Dict of embedded chunks of the reference transcripts and their participant IDs
    
    reference_evidence : string
        A string containing the reference evidence chunks and their PHQ8 scores
    
    Returns
    -------
    PHQ8ScoresWithExplanations
        Pydantic model containing PHQ-8 assessment results. Has 8 attributes 
        (PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired, PHQ8_Appetite, 
        PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving), each of type 
        PHQ8ScoreWithExplanation with 'evidence', 'reason', and 'score' fields.
    """
    quantitative_prompt = f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{patient_transcript}
</transcript>

{reference_evidence}

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{evidence, reason, score}} for little interest or pleasure in doing things (anhedonia)
- "PHQ8_Depressed": {{evidence, reason, score}} for feeling down, depressed, or hopeless (depressed mood)
- "PHQ8_Sleep": {{evidence, reason, score}} for trouble falling or staying asleep, or sleeping too much (sleep problems)
- "PHQ8_Tired": {{evidence, reason, score}} for feeling tired or having little energy (fatigue)
- "PHQ8_Appetite": {{evidence, reason, score}} for poor appetite or overeating (appetite/weight changes)
- "PHQ8_Failure": {{evidence, reason, score}} for feeling bad about yourself or that you are a failure (negative self-perception)
- "PHQ8_Concentrating": {{evidence, reason, score}} for trouble concentrating on things like reading or watching TV (concentration problems)
- "PHQ8_Moving": {{evidence, reason, score}} for moving or speaking slowly, or being fidgety/restless (psychomotor changes)"""

    system_prompt = """You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
8. Follow the format specified exactly"""

    response = requests.post(
        BASE_URL,
        json={
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": quantitative_prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_k": 1,
                "top_p": 1.0
            }
        }
    )

    try:
        response_data = response.json()
        content = response_data['message']['content']
        
        if '<answer>' in content and '</answer>' in content:
            content = content.split('<answer>')[1].split('</answer>')[0].strip()
        
        if content.startswith('```json'):
            content = content.split('```json')[1].split('```')[0].strip()
        elif content.startswith('```'):
            content = content.split('```')[1].split('```')[0].strip()
        
        scores_dict = json.loads(content)
        phq8_scores = PHQ8ScoresWithExplanations(**scores_dict)
        
        return phq8_scores
        
    except Exception as e:
        raise

def main():
    log_message("Starting PHQ-8 batch processing")
    
    # Load data
    log_message("Loading PHQ-8 ground truth data...")
    dev_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")
    train_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
    phq8_ground_truths = pd.concat([dev_split_phq8, train_split_phq8], ignore_index=True)
    phq8_ground_truths = phq8_ground_truths.sort_values('Participant_ID').reset_index(drop=True)
    
    # Load participant list
    unique_participants = [302, 303, 304, 305, 307, 310, 312, 313, 315, 316, 317, 318, 320, 321, 322, 324, 325, 326, 327, 328, 330, 331, 333, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355, 357, 358, 360, 362, 363, 364, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377, 379, 380, 381, 382, 383, 385, 386, 388, 389, 390, 391, 392, 393, 395, 397, 400, 401, 402, 403, 406, 409, 412, 413, 414, 415, 416, 417, 418, 419, 420, 423, 425, 426, 427, 428, 429, 430, 433, 434, 436, 437, 439, 440, 443, 444, 445, 446, 447, 448, 449, 451, 454, 455, 456, 457, 458, 459, 463, 464, 468, 471, 472, 473, 475, 476, 477, 478, 479, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492]
    
    # Get participants to process (starting from index 40)
    participants_to_process = unique_participants[40:]

    log_message(f"Total participants to process: {len(participants_to_process)}")
    
    # Load already processed participants
    processed_ids = load_processed_participants()
    log_message(f"Already processed participants: {len(processed_ids)}")
    
    # Load embeddings
    log_message("Loading participant embeddings...")
    pickle_file = r"/data/users2/agreene46/ai-psychiatrist/participant_embedded_transcripts.pkl"
    try:
        with open(pickle_file, 'rb') as f:
            participant_embedded_transcripts = pickle.load(f)
    except Exception as e:
        log_message(f"Error loading embeddings: {str(e)}")
        return
    
    # Process each participant
    successful = 0
    failed = 0
    
    for idx, participant_id in enumerate(participants_to_process):
        # Skip if already processed
        if participant_id in processed_ids:
            log_message(f"Skipping participant {participant_id} - already processed")
            continue
        
        log_message(f"Processing participant {participant_id} ({idx+1}/{len(participants_to_process)})...")
        
        try:
            # Load transcript
            transcript_path = fr"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/{participant_id}_P/{participant_id}_TRANSCRIPT.csv"

            current_transcript = pd.read_csv(transcript_path, sep="\t")
            
            # Handle missing values
            current_transcript['speaker'] = current_transcript['speaker'].fillna('Unknown').astype(str)
            current_transcript['value'] = current_transcript['value'].fillna('').astype(str)
            
            # Format transcript
            current_patient_transcript = '\n'.join(current_transcript['speaker'] + ': ' + current_transcript['value'])
            
            # Get ground truth for this participant
            participant_ground_truth = phq8_ground_truths.loc[
                phq8_ground_truths['Participant_ID'] == participant_id
            ]
            
            if participant_ground_truth.empty:
                log_message(f"No ground truth data found for participant {participant_id}")
                failed += 1
                continue
            
            # Extract evidence
            evidence_dict = extract_evidence_for_participant(current_patient_transcript, participant_id)
            
            # Generate reference evidence (excluding current participant from embeddings)
            reference_embeddings = {k: v for k, v in participant_embedded_transcripts.items() if k != participant_id}
            reference_evidence = process_evidence_for_references(
                evidence_dict, 
                reference_embeddings, 
                phq8_ground_truths
            )
            
            # Run PHQ-8 analysis
            phq8_scores = run_phq8_analysis_batch(
                current_patient_transcript, 
                participant_ground_truth, 
                reference_evidence
            )
            
            # Save results
            save_participant_result(participant_id, phq8_scores)
            
            successful += 1
            log_message(f"Successfully processed participant {participant_id}", print_to_console=False)
            
        except FileNotFoundError:
            error_msg = f"Transcript file not found for participant {participant_id}"
            log_message(f"ERROR: {error_msg}")
            failed += 1
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error for participant {participant_id}: {str(e)}"
            log_message(f"ERROR: {error_msg}")
            failed += 1
            
        except requests.RequestException as e:
            error_msg = f"API request failed for participant {participant_id}: {str(e)}"
            log_message(f"ERROR: {error_msg}")
            failed += 1
            
        except Exception as e:
            error_msg = f"Unexpected error for participant {participant_id}: {str(e)}"
            log_message(f"ERROR: {error_msg}")
            failed += 1
    
    # Final summary
    log_message(f"Batch processing completed. Successful: {successful}, Failed: {failed}")
    log_message(f"Results saved to: {JSONL_FILE}")
    log_message(f"Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()