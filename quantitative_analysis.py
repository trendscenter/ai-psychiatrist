import json
import requests
import pandas as pd
from pydantic import BaseModel
from typing import List
from typing import Union
import csv
from datetime import datetime

# Ollama Config
OLLAMA_NODE = "arctrdagn032"
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"

model = "gemma3-optimized:27b" # qwq:latest

############################################################################################################################
# Grabbing all participant IDs that have all the PHQ-8 questionare data #####################################################
#############################################################################################################################

dev_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")
train_split_phq8 = pd.read_csv(r"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")

participant_ids = set()

# Grabbing unique participant ID values and putting them in a list

participant_ids.update(dev_split_phq8['Participant_ID'])
participant_ids.update(train_split_phq8['Participant_ID'])

participant_list = sorted(list(participant_ids))
# Did this cus execution loop kept breaking and I was too lazy to fix
values_already_done = {302, 303, 304, 305, 307, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 331, 333, 335, 336, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 367, 368, 369, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 383, 385, 386, 388, 389, 390, 391, 392, 393, 395, 397, 400, 401, 402, 403, 404, 406, 409, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 425, 426, 427, 428, 429, 430, 433, 434, 436, 437, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449, 451, 454, 455, 456, 457, 458, 459, 463, 464, 468, 471, 472, 473, 474}
for value in values_already_done:
    if value in participant_list:
        participant_list.remove(value)
#print(participant_list)
#participant_list = [338]
#print(participant_list)

############################################################################################################################
# Quantitative Assessment ##################################################################################################
#############################################################################################################################

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
6. Follow the format specified exactly"""

def run_phq8_analysis(patient_transcript, ground_truth):
    quantitative_prompt = f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{patient_transcript}
</transcript>

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. Evaluate the frequency/severity based on available evidence
3. If no relevant evidence exists, mark as "N/A" rather than assuming absence
4. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including why N/A if applicable
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

    # Most deterministic temp, top_k, and top_p
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

    # Parse and validate the response
    try:
        response_data = response.json()
        content = response_data['message']['content']
        
        # Extract content from <answer> tags if present
        if '<answer>' in content and '</answer>' in content:
            content = content.split('<answer>')[1].split('</answer>')[0].strip()
        
        # Remove markdown code blocks if present
        if content.startswith('```json'):
            content = content.split('```json')[1].split('```')[0].strip()
        elif content.startswith('```'):
            content = content.split('```')[1].split('```')[0].strip()
        
        # Parse the JSON response and validate with Pydantic
        scores_dict = json.loads(content)
        phq8_scores = PHQ8ScoresWithExplanations(**scores_dict)
        
        # Extract the 8 PHQ-8 score values
        scores_list = [
            phq8_scores.PHQ8_NoInterest.score,
            phq8_scores.PHQ8_Depressed.score,
            phq8_scores.PHQ8_Sleep.score,
            phq8_scores.PHQ8_Tired.score,
            phq8_scores.PHQ8_Appetite.score,
            phq8_scores.PHQ8_Failure.score,
            phq8_scores.PHQ8_Concentrating.score,
            phq8_scores.PHQ8_Moving.score
        ]
        
        print("Comparison of Predicted vs Ground Truth:")
        print("Metric\t\t\tPredicted\tGround Truth\tDifference")
        print("-" * 65)

        differences = []
        n_available = 0
        num_questions_NA = 0
        metrics = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 
                'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']
        predicted_values = [phq8_scores.PHQ8_NoInterest.score, phq8_scores.PHQ8_Depressed.score, phq8_scores.PHQ8_Sleep.score, 
                            phq8_scores.PHQ8_Tired.score, phq8_scores.PHQ8_Appetite.score, phq8_scores.PHQ8_Failure.score,
                            phq8_scores.PHQ8_Concentrating.score, phq8_scores.PHQ8_Moving.score]

        for metric, pred_val in zip(metrics, predicted_values):
            gt_val = int(ground_truth[metric])
            if pred_val == "N/A":
                diff_str = "N/A"
                num_questions_NA += 1
            else:
                pred_val_int = int(pred_val)
                diff = abs(pred_val_int - gt_val)
                differences.append(diff)
                diff_str = str(diff)
                n_available += 1

        # Calculate metrics
        if n_available > 0:
            avg_difference = sum(differences) / n_available
            accuracy_on_available = 1 - (avg_difference / 3)
        else:
            avg_difference = float('inf')
            accuracy_on_available = 0
        
        # Accuracy * % available questions
        overall_accuracy = accuracy_on_available * (1 - (num_questions_NA / 8))
        
        print("-" * 65)
        if n_available > 0:
            print(f"Average Absolute Difference (on available): {avg_difference:.2f}")
            print(f"Accuracy on available questions: {accuracy_on_available:.2%}")
        print(f"Questions marked N/A: {num_questions_NA}/8")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        
        # Reasoning and evidence section
        print("\n\nDetailed Reasoning for Each Score:")
        print("=" * 80)
        
        symptom_names = {
            'PHQ8_NoInterest': 'Little Interest/Pleasure',
            'PHQ8_Depressed': 'Feeling Depressed',
            'PHQ8_Sleep': 'Sleep Problems',
            'PHQ8_Tired': 'Fatigue',
            'PHQ8_Appetite': 'Appetite Changes',
            'PHQ8_Failure': 'Negative Self-Perception',
            'PHQ8_Concentrating': 'Concentration Problems',
            'PHQ8_Moving': 'Psychomotor Changes'
        }
        
        for key, symptom_name in symptom_names.items():
            score_data = getattr(phq8_scores, key)
            print(f"\n{symptom_name} (Score: {score_data.score})")
            print("-" * 40)
            print(f"Evidence: {score_data.evidence}")
            print(f"Reason: {score_data.reason}")

        return phq8_scores, avg_difference, accuracy_on_available, num_questions_NA, overall_accuracy

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing response: {e}")
        print("Raw response:", response)
        print(json.dumps(response.json(), indent=2))
        return None, None, None, None, None

############################################################################################################################
# Execution Loop (Slightly broken) #########################################################################################
#############################################################################################################################

if __name__ == "__main__":

    # Specify output path
    csv_file = f"/data/users2/agreene46/ai-psychiatrist/analysis_output/results.csv"
    json_file = f"/data/users2/agreene46/ai-psychiatrist/analysis_output/results_detailed.jsonl"

    # Initialize CSV file with headers
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['participant_id', 'timestamp', 'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 
                        'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving',
                        'avg_difference', 'accuracy_on_available', 'num_questions_na', 'overall_accuracy'])

    # Execution loop
    for participant_id in participant_list:
        current_transcript = pd.read_csv(fr"/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset/{participant_id}_P/{participant_id}_TRANSCRIPT.csv", sep="\t")
        
        # Reformatting transcript data to be a string with speaker name + text
        current_patient_transcript = '\n'.join(current_transcript['speaker'].astype(str) + ': ' + current_transcript['value'].astype(str))        
        
        # Get ground truth data for this participant
        if participant_id in train_split_phq8['Participant_ID'].values:
            ground_truth = train_split_phq8[train_split_phq8['Participant_ID'] == participant_id].iloc[0]
        else:
            ground_truth = dev_split_phq8[dev_split_phq8['Participant_ID'] == participant_id].iloc[0]

        # Run analysis
        phq8_scores, avg_difference, accuracy_on_available, num_questions_NA, overall_accuracy = run_phq8_analysis(current_patient_transcript, ground_truth)

        if phq8_scores is not None:
            # Save to CSV
            timestamp = datetime.now().isoformat()
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    participant_id, timestamp,
                    phq8_scores.PHQ8_NoInterest.score, phq8_scores.PHQ8_Depressed.score, phq8_scores.PHQ8_Sleep.score,
                    phq8_scores.PHQ8_Tired.score, phq8_scores.PHQ8_Appetite.score, phq8_scores.PHQ8_Failure.score,
                    phq8_scores.PHQ8_Concentrating.score, phq8_scores.PHQ8_Moving.score,
                    avg_difference, accuracy_on_available, num_questions_NA, overall_accuracy
                ])
            
            # Save detailed data to JSON Lines
            detailed_data = {
                "participant_id": participant_id,
                "timestamp": timestamp,
                "PHQ8_NoInterest": {"evidence": phq8_scores.PHQ8_NoInterest.evidence, "reason": phq8_scores.PHQ8_NoInterest.reason, "score": phq8_scores.PHQ8_NoInterest.score},
                "PHQ8_Depressed": {"evidence": phq8_scores.PHQ8_Depressed.evidence, "reason": phq8_scores.PHQ8_Depressed.reason, "score": phq8_scores.PHQ8_Depressed.score},
                "PHQ8_Sleep": {"evidence": phq8_scores.PHQ8_Sleep.evidence, "reason": phq8_scores.PHQ8_Sleep.reason, "score": phq8_scores.PHQ8_Sleep.score},
                "PHQ8_Tired": {"evidence": phq8_scores.PHQ8_Tired.evidence, "reason": phq8_scores.PHQ8_Tired.reason, "score": phq8_scores.PHQ8_Tired.score},
                "PHQ8_Appetite": {"evidence": phq8_scores.PHQ8_Appetite.evidence, "reason": phq8_scores.PHQ8_Appetite.reason, "score": phq8_scores.PHQ8_Appetite.score},
                "PHQ8_Failure": {"evidence": phq8_scores.PHQ8_Failure.evidence, "reason": phq8_scores.PHQ8_Failure.reason, "score": phq8_scores.PHQ8_Failure.score},
                "PHQ8_Concentrating": {"evidence": phq8_scores.PHQ8_Concentrating.evidence, "reason": phq8_scores.PHQ8_Concentrating.reason, "score": phq8_scores.PHQ8_Concentrating.score},
                "PHQ8_Moving": {"evidence": phq8_scores.PHQ8_Moving.evidence, "reason": phq8_scores.PHQ8_Moving.reason, "score": phq8_scores.PHQ8_Moving.score}
            }

            with open(json_file, 'a') as f:
                f.write(json.dumps(detailed_data) + '\n')
            
            print(f"\nCompleted analysis for participant {participant_id}")
        else:
            print(f"\nFailed to analyze participant {participant_id} - skipping")
        
        print("="*80)