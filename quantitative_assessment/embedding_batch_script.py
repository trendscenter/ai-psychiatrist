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
from sklearn.model_selection import train_test_split
import argparse
import signal
from contextlib import contextmanager
import math

############################################################################################################################
##################### Configuring ollama, paths, and pydantic model s######################################################
############################################################################################################################

class PHQ8ScoreWithExplanation(BaseModel):
    evidence: str  # Direct quotes or references from interview
    reason: str  # Brief explanation for the score
    cosine_similarity: str = "N/A"  # Comma-separated cosine similarity scores, default to N/A
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

def get_embedding(text, model="dengcao/Qwen3-Embedding-8B:Q8_0", dim=None):
    """
    Creates embedding from given text input and model 

    Parameters
    ----------
    text : string
        The text to be embedded
    model : string
        The name of the ollama model to be used for embedding
    dim : int, optional
        If provided, truncate to this dimension and normalize (MRL support)

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
        embedding = response.json()["embedding"]
        
        # Manually setting dimension because ollama doesn't natively support atm
        if dim is not None:
            # Truncate and normalize for MRL models
            embedding = embedding[:dim]
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
        
        return embedding
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

def find_similar_chunks(evidence_text_embedding, participant_embedded_transcripts, top_k):
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

def load_processed_participants(jsonl_file):
    """
    Loads all participants ids from the jsonl file so as to skip them when processing

    Returns
    -------
    set
        All processed participant id ints
    """
    processed_ids = set()
    
    if os.path.exists(jsonl_file):
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed_ids.add(data['participant_id'])
                    except:
                        continue
        except Exception as e:
            log_message(f"Could not read existing JSONL file: {str(e)}")
    
    return processed_ids

def save_participant_result(participant_id, phq8_scores, output_file):
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
        "evidence", "reason", "cosine_similarity", and "score"
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
            "cosine_similarity": getattr(score_data, 'cosine_similarity', 'N/A'),
            "score": score_data.score
        }
    
    # Append to JSONL file
    with open(output_file, 'a') as f:
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
                    # Fairly deterministic parameters
                    "temperature": 0.2,
                    "top_k": 20,
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

def process_evidence_for_references(evidence_dict, participant_embedded_transcripts, phq8_ground_truths, dim=None):
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
    tuple
        A tuple containing:
        - str: A string with all the reference transcripts chunks and their corresponding PHQ8 scores
        - dict: Dictionary mapping PHQ8 questions to their cosine similarity scores
    """

    evidence_keys = [
        'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
        'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
    ]
    
    all_references = []
    similarity_scores = {}
    
    for evidence_key in evidence_keys:
        # Get evidence texts for this key
        evidence_texts = evidence_dict.get(evidence_key, [])
        
        # Initialize empty similarity scores for this key
        similarity_scores[evidence_key] = []
        
        # Skip if empty
        if not evidence_texts:
            continue
        
        # Combine evidence text into single string
        combined_text = '\n'.join(evidence_texts)
        
        # Skip if less than 15 characters
        if len(combined_text) < 15:
            continue
        
        log_message(f"Processing {evidence_key}...")
        
        try:
            # Get embedding for combined evidence text
            evidence_embedding = get_embedding(combined_text, dim=dim)
            
            # Find top_k similar chunks
            similar_chunks = find_similar_chunks(
                evidence_embedding, 
                participant_embedded_transcripts, 
                top_k=int(examples_num)
            )
            
            # Extract similarity scores for this evidence key
            similarity_scores[evidence_key] = [str(chunk['similarity']) for chunk in similar_chunks]
            
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
    
    return reference_evidence, similarity_scores

################################################################################################################################
############ Running quantitative analysis, but with reference transcripts this time. Grabs evidence, reasons, and outputs score
#################################################################################################################################


def run_phq8_analysis_batch(patient_transcript, reference_evidence="", similarity_scores=None):
    """
    Grabs chunks from other transcripts that are similar to the evidence pulled for the current transcript.
    Then, grabs those chunks ground truth scores and formats all that into a string for use in the prompt.

    Parameters
    ----------
    patient_transcript : string
        The patients transcript

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
                # Fairly deterministic parameters
                "temperature": 0.2,
                "top_k": 20,
                "top_p": 0.8
            }
        }
    )
    try:
        # Parsing response
        response_data = response.json()
        content = response_data['message']['content']
        
        if '<answer>' in content and '</answer>' in content:
            content = content.split('<answer>')[1].split('</answer>')[0].strip()
        
        if content.startswith('```json'):
            content = content.split('```json')[1].split('```')[0].strip()
        elif content.startswith('```'):
            content = content.split('```')[1].split('```')[0].strip()
        
        scores_dict = json.loads(content)
        
        # Add cosine similarity scores to each PHQ8 question for output
        if similarity_scores:
            for phq_key in scores_dict.keys():
                if phq_key in similarity_scores and similarity_scores[phq_key]:
                    # Convert list of similarity scores to comma-separated string
                    similarity_string = ",".join(similarity_scores[phq_key])
                    scores_dict[phq_key]["cosine_similarity"] = similarity_string
                else:
                    # If no similarity scores available, set to empty string or N/A
                    scores_dict[phq_key]["cosine_similarity"] = "N/A"
        else:
            # If no similarity_scores provided, add N/A to all questions
            for phq_key in scores_dict.keys():
                scores_dict[phq_key]["cosine_similarity"] = "N/A"
        
        phq8_scores = PHQ8ScoresWithExplanations(**scores_dict)
        
        return phq8_scores
        
    except Exception as e:
        raise

def create_score_ranges(score):
    """
    Creates PHQ8 score ranges based on severity

    Parameters
    ----------
    score : int
        PHQ8 total score

    Returns
    -------
    str
        String of the depression severity
    """
    if score <= 4:
        return 'minimal'  # 0-4: Minimal depression
    elif score <= 9:
        return 'mild'     # 5-9: Mild depression
    elif score <= 14:
        return 'moderate' # 10-14: Moderate depression
    elif score <= 19:
        return 'mod_severe' # 15-19: Moderately severe
    else:
        return 'severe'   # 20-24: Severe depression

def check_balance(ids, df, split_name):
    """
    Checks the stratification balance

    Parameters
    ----------
    ids : list
        list of participant ids
    df : pandas dataframe
        dataframe of all the participant ground truth info
    split_name : string
        name of the split

    Prints
    -------
    string
        Strings giving information about the balance of the data
    """
    subset = df[df['Participant_ID'].isin(ids)]
    log_message(f"\n{split_name} set balance:")
    log_message(f"PHQ8_Binary: {subset['PHQ8_Binary'].value_counts().to_dict()}")
    log_message(f"Gender: {subset['Gender'].value_counts().to_dict()}")
    
    # Show PHQ8_Score distribution
    score_dist = subset['PHQ8_Score'].value_counts().sort_index()
    log_message(f"PHQ8_Score distribution: {dict(score_dist)}")
    log_message(f"Score stats: Mean={subset['PHQ8_Score'].mean():.1f}, Std={subset['PHQ8_Score'].std():.1f}")
    
    # Show how many unique categories are in this split
    unique_categories = subset['strat_var'].nunique()
    log_message(f"Unique categories (Gender_PHQ8Score): {unique_categories}")

@contextmanager
def timeout(duration):
    """
    Handles length of time each participant has to process.
    Times out when duration has passed.

    Parameters
    ----------
    duration : int
        the time, in seconds, a participant can process
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


if __name__ == "__main__":

    # Setting commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_step', help='Comma-separated list of chunk steps')
    parser.add_argument('--examples_num', help='Comma-separated list of example numbers')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to run each chunk_step and examples_num combination')
    parser.add_argument('--dims', help='Comma-separated list of dim numbers')
    parser.add_argument('--ollama_node')
    args = parser.parse_args()

    # Parse comma-separated lists
    chunk_steps = [s.strip() for s in args.chunk_step.split(',')]
    examples_nums = [s.strip() for s in args.examples_num.split(',')]
    if args.dims:
        dims = [s.strip() for s in args.dims.split(',')]
    else:
        dims = ['']
    
    # Ollama Config
    OLLAMA_NODE = args.ollama_node
    BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
    model = "gemma3-optimized:27b" #gpt-oss:20b

    # Output paths
    OUTPUT_DIR = r"/data/users2/agreene46/ai-psychiatrist/analysis_output"
    LOG_FILE = os.path.join(OUTPUT_DIR, "embedding_log_file.txt")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Log initial configuration
    log_message("="*80)
    log_message("Starting PHQ-8 batch processing with multiple configurations")
    log_message(f"Chunk steps to process: {chunk_steps}")
    log_message(f"Example numbers to process: {examples_nums}")
    log_message(f"Dims to process: {dims if args.dims else 'None specified'}")
    log_message(f"Number of runs per combination: {args.num_runs}")
    log_message(f"Total combinations to run: {len(chunk_steps) * len(dims) * len(examples_nums) * args.num_runs}")
    log_message("="*80)
    
    # Load data
    log_message("Loading PHQ-8 ground truth data...")
    dev_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")
    train_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
    phq8_ground_truths = pd.concat([dev_split_phq8, train_split_phq8], ignore_index=True)
    phq8_ground_truths = phq8_ground_truths.sort_values('Participant_ID').reset_index(drop=True)
    
    phq8_ground_truths['score_range'] = phq8_ground_truths['PHQ8_Score'].apply(create_score_ranges)
    log_message(f"PHQ8_Score ranges distribution:\n{phq8_ground_truths['score_range'].value_counts()}")

    # Stratify data based on Gender + PHQ8_Score
    phq8_ground_truths['strat_var'] = (phq8_ground_truths['Gender'].astype(str) + '_' + 
                                    phq8_ground_truths['PHQ8_Score'].astype(str))

    log_message(f"Stratification groups (Gender_PHQ8Score):\n{phq8_ground_truths['strat_var'].value_counts().sort_index()}")

    # Getting category sizes
    strat_counts = phq8_ground_truths['strat_var'].value_counts()
    log_message(f"\nCategory distribution:")
    log_message(f"Categories with >= 3 subjects: {(strat_counts >= 3).sum()}")
    log_message(f"Categories with 2 subjects: {(strat_counts == 2).sum()}")
    log_message(f"Categories with 1 subject: {(strat_counts == 1).sum()}")

    # Separate participants into different groups based on category size
    categories_gte3 = strat_counts[strat_counts >= 3].index.tolist()
    categories_eq2 = strat_counts[strat_counts == 2].index.tolist()
    categories_eq1 = strat_counts[strat_counts == 1].index.tolist()

    # Get participant IDs for each group
    participants_gte3 = phq8_ground_truths[phq8_ground_truths['strat_var'].isin(categories_gte3)]
    participants_eq2 = phq8_ground_truths[phq8_ground_truths['strat_var'].isin(categories_eq2)]
    participants_eq1 = phq8_ground_truths[phq8_ground_truths['strat_var'].isin(categories_eq1)]

    train_ids = []
    val_ids = []
    test_ids = []

    # Process categories with >= 3 subjects using sklearn
    if len(participants_gte3) > 0:
        # Group by category to handle small categories specially
        for category in categories_gte3:
            category_participants = participants_gte3[participants_gte3['strat_var'] == category]['Participant_ID'].tolist()
            n = len(category_participants)
            
            # Calculate ideal splits
            ideal_train = n * 0.40
            ideal_val = n * 0.30
            ideal_test = n * 0.30
            
            # Use consistent 40/30/30 split for all categories >= 3
            actual_train = round(ideal_train)
            actual_val = round(ideal_val) 
            actual_test = n - actual_train - actual_val

            # Ensure no negative values
            if actual_test < 0:
                actual_train = max(1, actual_train - 1)
                actual_test = n - actual_train - actual_val
            
            # Randomly assign participants
            np.random.seed(42)
            shuffled = category_participants.copy()
            np.random.shuffle(shuffled)
            
            train_ids.extend(shuffled[:actual_train])
            val_ids.extend(shuffled[actual_train:actual_train+actual_val])
            test_ids.extend(shuffled[actual_train+actual_val:])
            
    # Process categories with exactly 2 subjects (1 for train, 1 for test)
    if len(participants_eq2) > 0:
        for category in categories_eq2:
            category_participants = participants_eq2[participants_eq2['strat_var'] == category]['Participant_ID'].tolist()
            # Randomly assign 1 to train and 1 to test
            np.random.shuffle(category_participants)
            train_ids.append(category_participants[0])
            test_ids.append(category_participants[1])
        
        log_message(f"\nProcessed {len(participants_eq2)} participants from categories with 2 subjects")
        log_message(f"  {len(categories_eq2)} categories: 1 to train, 1 to test each")
        
        log_message(f"\nProcessed {len(participants_eq2)} participants from categories with 2 subjects")
        log_message(f"  {len(categories_eq2)} categories: 1 to train, 1 to validation each")

    # Process categories with exactly 1 subject (all go to train)
    if len(participants_eq1) > 0:
        train_ids.extend(participants_eq1['Participant_ID'].tolist())
        log_message(f"\nProcessed {len(participants_eq1)} participants from categories with 1 subject")
        log_message(f"  All {len(participants_eq1)} assigned to train set")

    # Post-Prossessing Override: Handle cases where PHQ8_Score has exactly 2 participants (regardless of gender) after previous stratification takes place
    phq8_score_counts = phq8_ground_truths['PHQ8_Score'].value_counts()
    scores_with_2 = phq8_score_counts[phq8_score_counts == 2].index.tolist()

    for score in scores_with_2:
        score_participants = phq8_ground_truths[phq8_ground_truths['PHQ8_Score'] == score]['Participant_ID'].tolist()
        
        # Remove these participants from their current assignments
        for pid in score_participants:
            if pid in train_ids:
                train_ids.remove(pid)
            if pid in val_ids:
                val_ids.remove(pid)
            if pid in test_ids:
                test_ids.remove(pid)
        
        # Reassign: 1 to validation, 1 to test
        np.random.shuffle(score_participants)
        val_ids.append(score_participants[0])
        test_ids.append(score_participants[1])
        
        log_message(f"Override applied for PHQ8_Score={score}: 1 to validation, 1 to test")

    log_message(f"\nFinal split sizes:")
    log_message(f"Train: {len(train_ids)} ({len(train_ids)/len(phq8_ground_truths)*100:.1f}%)")
    log_message(f"Validation: {len(val_ids)} ({len(val_ids)/len(phq8_ground_truths)*100:.1f}%)")
    log_message(f"Test: {len(test_ids)} ({len(test_ids)/len(phq8_ground_truths)*100:.1f}%)")
    log_message(f"Total: {len(train_ids) + len(val_ids) + len(test_ids)}")

    check_balance(train_ids, phq8_ground_truths, "Train")
    check_balance(val_ids, phq8_ground_truths, "Validation")
    check_balance(test_ids, phq8_ground_truths, "Test")

    # Show category distribution across splits
    log_message("\nDetailed category distribution:")
    for split_name, split_ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        subset = phq8_ground_truths[phq8_ground_truths['Participant_ID'].isin(split_ids)]
        category_counts = subset['strat_var'].value_counts()
        log_message(f"\n{split_name} categories with counts:")
        for cat in sorted(category_counts.index):
            gender, score = cat.split('_')
            log_message(f"  Gender={gender}, PHQ8={score}: {category_counts[cat]} subjects")

    # Output the three data splits
    log_message(f"\nTrain IDs ({len(train_ids)}): {train_ids}")
    log_message(f"\nValidation IDs ({len(val_ids)}): {val_ids}")
    log_message(f"\nTest IDs ({len(test_ids)}): {test_ids}")

    # Track overall statistics
    overall_stats = {
        'total_combinations': len(chunk_steps) * len(dims) * len(examples_nums) * args.num_runs,
        'completed_combinations': 0,
        'failed_combinations': 0
    }
    # Process each combination of chunk_step and examples_num
    combination_num = 0
    for chunk_step in chunk_steps:
        for dim in dims:
            # Load embeddings
            if dim:
                pickle_file = fr"/data/users2/agreene46/ai-psychiatrist/{chunk_step}_dim_{str(dim)}_participant_embedded_transcripts.pkl"
            else:
                pickle_file = fr"/data/users2/agreene46/ai-psychiatrist/{chunk_step}_participant_embedded_transcripts.pkl"

            log_message(f"Loading participant embeddings{' with dim=' + dim if dim else ''}...")
            try:
                with open(pickle_file, 'rb') as f:
                    participant_embedded_transcripts = pickle.load(f)
            except Exception as e:
                log_message(f"Error loading embeddings from {pickle_file}: {str(e)}")
                raise
                
            for examples_num in examples_nums:
                for run_num in range(1, args.num_runs + 1):
                    combination_num += 1
                    
                    # Log current combination
                    log_message("\n" + "="*80)
                    log_message(f"PROCESSING COMBINATION {combination_num}/{overall_stats['total_combinations']}")
                    log_message(f"chunk_step: {chunk_step}, dim: {dim if dim else 'default'}, examples_num: {examples_num}, run: {run_num}")  # Fix: add dim
                    log_message("="*80)
                    if dim:
                        JSONL_FILE = os.path.join(OUTPUT_DIR, f"{chunk_step}_dim_{dim}_examples_{examples_num}_embedding_results_analysis_{run_num}.jsonl")
                    else:
                        JSONL_FILE = os.path.join(OUTPUT_DIR, f"{chunk_step}_examples_{examples_num}_embedding_results_analysis_{run_num}.jsonl")
                    log_message(f"Output file: {JSONL_FILE}")
                    log_message(f"Total participants to process: {len(val_ids)}")
                    
                    # Load already processed participants for this specific combination
                    processed_ids = load_processed_participants(JSONL_FILE)
                    log_message(f"Already processed participants for this combination: {len(processed_ids)}")
                    # Process each participant with retry logic
                    successful = 0
                    failed = 0
                    
                    # Runs analysis for each participant in the given set, change 'test_ids' to 'val_ids' to run validation
                    for idx, participant_id in enumerate(test_ids):
                        # Skip if already processed for this combination
                        if participant_id in processed_ids:
                            log_message(f"[{chunk_step}, {examples_num}, run {run_num}] Skipping participant {participant_id} - already processed")
                            continue
                        
                        # Retry logic - up to 10 attempts
                        max_retries = 10
                        retry_count = 0
                        participant_processed = False
                        
                        while retry_count < max_retries and not participant_processed:
                            retry_count += 1
                            
                            if retry_count > 1:
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] Retry attempt {retry_count}/{max_retries} for participant {participant_id}")
                            else:
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] Processing participant {participant_id} ({idx+1}/{len(test_ids)})...")
                            
                            try:
                                with timeout(600):  # 10 minutes of processing, afterwards the attempt is re-tried. Done to handle infinite looping
                                    # Load transcript
                                    transcript_path = fr"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/{participant_id}_P/{participant_id}_TRANSCRIPT.csv"
                                    current_transcript = pd.read_csv(transcript_path, sep="\t")
                                    
                                    # Handle missing values
                                    current_transcript['speaker'] = current_transcript['speaker'].fillna('Unknown').astype(str)
                                    current_transcript['value'] = current_transcript['value'].fillna('').astype(str)
                                    
                                    # Format transcript
                                    current_patient_transcript = '\n'.join(current_transcript['speaker'] + ': ' + current_transcript['value'])
        
                                    # Extract evidence (using current chunk_step and examples_num)
                                    evidence_dict = extract_evidence_for_participant(
                                        current_patient_transcript, 
                                        participant_id
                                    )
                                    
                                    # Generate reference evidence (excluding current participant from embeddings)
                                    reference_embeddings = {k: v for k, v in participant_embedded_transcripts.items() if k != participant_id}
                                    reference_evidence, similarity_scores = process_evidence_for_references(
                                        evidence_dict, 
                                        reference_embeddings, 
                                        phq8_ground_truths,
                                        dim=dim
                                    )

                                    # Run PHQ-8 analysis
                                    phq8_scores = run_phq8_analysis_batch(
                                        current_patient_transcript, 
                                        reference_evidence,
                                        similarity_scores 
                                    )
                                    
                                    # Save results to the combination-specific file
                                    save_participant_result(participant_id, phq8_scores, output_file=JSONL_FILE)
                                    
                                    successful += 1
                                    participant_processed = True
                                    log_message(f"[{chunk_step}, {examples_num}, run {run_num}] Successfully processed participant {participant_id} (attempt {retry_count})", print_to_console=False)
                            
                            except TimeoutError:
                                error_msg = f"Timeout: Participant {participant_id} took longer than 5 minutes (attempt {retry_count}/{max_retries})"
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] ERROR: {error_msg}")
                                
                                if retry_count >= max_retries:
                                    failed += 1
                                    log_message(f"[{chunk_step}, {examples_num}, run {run_num}] FAILED: Participant {participant_id} failed after {max_retries} attempts")
                                
                            except FileNotFoundError:
                                error_msg = f"Transcript file not found for participant {participant_id}"
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] ERROR: {error_msg}")
                                failed += 1
                                participant_processed = True  # Don't retry if file doesn't exist
                                
                            except json.JSONDecodeError as e:
                                error_msg = f"JSON parsing error for participant {participant_id}: {str(e)} (attempt {retry_count}/{max_retries})"
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] ERROR: {error_msg}")
                                
                                if retry_count >= max_retries:
                                    failed += 1
                                    log_message(f"[{chunk_step}, {examples_num}, run {run_num}] FAILED: Participant {participant_id} failed after {max_retries} attempts")
                                
                            except requests.RequestException as e:
                                error_msg = f"API request failed for participant {participant_id}: {str(e)} (attempt {retry_count}/{max_retries})"
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] ERROR: {error_msg}")
                                
                                if retry_count >= max_retries:
                                    failed += 1
                                    log_message(f"[{chunk_step}, {examples_num}, run {run_num}] FAILED: Participant {participant_id} failed after {max_retries} attempts")
                                
                            except Exception as e:
                                error_msg = f"Unexpected error for participant {participant_id}: {str(e)} (attempt {retry_count}/{max_retries})"
                                log_message(f"[{chunk_step}, {examples_num}, run {run_num}] ERROR: {error_msg}")
                                
                                if retry_count >= max_retries:
                                    failed += 1
                                    log_message(f"[{chunk_step}, {examples_num}, run {run_num}] FAILED: Participant {participant_id} failed after {max_retries} attempts")
                    
                    # Summary for this combination
                    log_message(f"\n[{chunk_step}, dim={dim if dim else 'default'}, {examples_num}, run {run_num}] Combination completed. Successful: {successful}, Failed: {failed}")
                    log_message(f"[{chunk_step}, dim={dim if dim else 'default'}, {examples_num}, run {run_num}] Results saved to: {JSONL_FILE}")
                    
                    if failed == 0:
                        overall_stats['completed_combinations'] += 1
                    else:
                        overall_stats['failed_combinations'] += 1
            
# Final overall summary
log_message("\n" + "="*80)
log_message("OVERALL BATCH PROCESSING COMPLETED")
log_message(f"Total combinations processed: {overall_stats['total_combinations']}")
log_message(f"Fully successful combinations: {overall_stats['completed_combinations']}")
log_message(f"Combinations with failures: {overall_stats['failed_combinations']}")
log_message(f"Log saved to: {LOG_FILE}")
log_message("="*80)