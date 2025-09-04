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
OLLAMA_NODE = "arctrdagn032" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use

#All Ids 
ID_NUM = [
    302, 304, 305, 307, 310, 313, 317, 318, 319, 320, 326, 335, 338, 339, 343, 346, 347, 351, 353, 357, 358, 360, 362, 364, 369, 371, 377, 383, 388, 389, 390, 392, 395, 397, 400, 402, 403, 409, 412, 415, 416, 423, 429, 441, 443, 445, 446, 448, 454, 457, 464, 474, 475, 479, 485, 486, 487
]

# Input file 
input_csv_path = "/data/users2/nblair7/analysis_results/GEMMAAPPENDEDASSESSMENTS.csv"  

#Output files
feedback_assessments_csv = "/data/users2/nblair7/analysis_results/qual_reassessment_GEMMA20.csv"  # re-evaluated qualitative assessments
feedback_evaluations_csv = "/data/users2/nblair7/analysis_results/qual_scores_GEMMA20.csv"  # re-evaluated evaluation scores

print(f"Input file: {input_csv_path}")
print(f"Failed IDs to process: {ID_NUM}")
print(f"Feedback assessments file: {feedback_assessments_csv}")
print(f"Feedback evaluations file: {feedback_evaluations_csv}")

# Load the CSV file
df = pd.read_csv(input_csv_path)
print(f"Loaded {len(df)} participants")

if ID_NUM:
    df = df[df['participant_id'].astype(str).isin([str(pid) for pid in ID_NUM])]
    print(f" {len(df)} participants to reprocess")
else:

    exit(1)

feedback_assessments = []  # Store re-evaluated qualitative assessments
feedback_evaluations = []  # Store re-evaluated evaluation scores
processed_count = 0
skipped_count = 0
feedback_count = 0
failed_evaluations = []

# Check for existing feedback files to resume processing
completed_subjects = set()
if os.path.exists(feedback_assessments_csv):
    existing_feedback_assessments = pd.read_csv(feedback_assessments_csv)
    completed_subjects.update(existing_feedback_assessments['participant_id'].tolist())
    feedback_assessments = existing_feedback_assessments.to_dict('records')
    print(f"Found existing feedback assessments: {len(feedback_assessments)} records")

if os.path.exists(feedback_evaluations_csv):
    existing_feedback_evaluations = pd.read_csv(feedback_evaluations_csv)
    completed_subjects.update(existing_feedback_evaluations['participant_id'].tolist())
    feedback_evaluations = existing_feedback_evaluations.to_dict('records')
    print(f"Found existing feedback evaluations: {len(feedback_evaluations)} records")

if completed_subjects:
    print(f"Already completed {len(completed_subjects)} subjects")
    df = df[~df['participant_id'].isin(completed_subjects)]
    print(f"Remaining subjects to process: {len(df)}")

for index, row in df.iterrows():
    participant_id = row['participant_id']
    qualitative_assessment = str(row['qualitative_assessment']) if 'qualitative_assessment' in row else ""
    qualitative_assessment = qualitative_assessment.replace("\n", " ")
    
    print(f"\n--- Processing {index + 1}/{len(df)}: {participant_id} ---")
    
    # Load transcript for this participant
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
        
    except Exception as e:
        print(f"Error loading transcript for {participant_id}: {e}")
        skipped_count += 1
        continue
    
    start_time = time.time()
    
    # Original evaluation prompts (for initial assessment to determine if feedback is needed)
    coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only. Compare qualitative_assessment to the provided transcript. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.
Score from 1-5 solely based on the following criteria: 
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
*NO FLOATING POINT SCORES, ONLY INTEGERS 1-5*
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
*NO FLOATING POINT SCORES, ONLY INTEGERS 1-5*
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
*NO FLOATING POINT SCORES, ONLY INTEGERS 1-5*
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
*NO FLOATING POINT SCORES, ONLY INTEGERS 1-5*
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

    # Initial evaluation requests
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
        # Initial evaluation to check if feedback is needed
        initial_scores = {}
        initial_explanations = {}
        
        # coherence
        print("  Getting initial coherence response...")
        coherence_response = requests.post(BASE_URL, json=coherence_request, timeout=timeout-10)
        if coherence_response.status_code == 200:
            coherence_content = coherence_response.json()['message']['content']
            coherence_score, _ = parse_score_and_explanation(coherence_content)
            initial_scores['coherence'] = coherence_score
            initial_explanations['coherence'] = coherence_content
            print(f"  Initial coherence score: {coherence_score}")
        else:
            initial_scores['coherence'] = None
            initial_explanations['coherence'] = None
        
        time.sleep(2)
        
        # completeness
        print("  Getting initial completeness response...")
        completeness_response = requests.post(BASE_URL, json=completeness_request, timeout=timeout-10)
        if completeness_response.status_code == 200:
            completeness_content = completeness_response.json()['message']['content']
            completeness_score, _ = parse_score_and_explanation(completeness_content)
            initial_scores['completeness'] = completeness_score
            initial_explanations['completeness'] = completeness_content
            print(f"  Initial completeness score: {completeness_score}")
        else:
            initial_scores['completeness'] = None
            initial_explanations['completeness'] = None
        
        time.sleep(2)
        
        # specificity
        print("  Getting initial specificity response...")
        specificity_response = requests.post(BASE_URL, json=specificity_request, timeout=timeout-10)
        if specificity_response.status_code == 200:
            specificity_content = specificity_response.json()['message']['content']
            specificity_score, _ = parse_score_and_explanation(specificity_content)
            initial_scores['specificity'] = specificity_score
            initial_explanations['specificity'] = specificity_content
            print(f"  Initial specificity score: {specificity_score}")
        else:
            initial_scores['specificity'] = None
            initial_explanations['specificity'] = None
        
        time.sleep(2)
        
        # accuracy
        print("  Getting initial accuracy response...")
        accuracy_response = requests.post(BASE_URL, json=accuracy_request, timeout=timeout-10)
        if accuracy_response.status_code == 200:
            accuracy_content = accuracy_response.json()['message']['content']
            accuracy_score, _ = parse_score_and_explanation(accuracy_content)
            initial_scores['accuracy'] = accuracy_score
            initial_explanations['accuracy'] = accuracy_content
            print(f"  Initial accuracy score: {accuracy_score}")
        else:
            initial_scores['accuracy'] = None
            initial_explanations['accuracy'] = None
        
        time.sleep(2)
        
        # Check if any score is <= 3 to trigger feedback loop
        low_scores = []
        for metric in ['coherence', 'completeness', 'specificity', 'accuracy']:
            if initial_scores[metric] and initial_scores[metric] <= 2:
                low_scores.append(metric)
        
        if low_scores:
            print(f"  LOW SCORES DETECTED: {low_scores} - Starting enhanced feedback loop...")
            feedback_count += 1
            
            # Keep track of all attempts for this participant
            iteration = 0
            max_iterations = 20  # Prevent infinite loops
            current_assessment = qualitative_assessment
            current_scores = initial_scores.copy()
            current_explanations = initial_explanations.copy()
            all_low_scores = []  # Track which scores were low across iterations
            
            # Feedback loop - keep trying until all scores are 4 or 5
            while low_scores and iteration < max_iterations:
                iteration += 1
                print(f"    Iteration {iteration}: Improving assessment for low scores: {low_scores}")
                all_low_scores.extend(low_scores)
                
                # Build feedback message for low scoring metrics
                feedback_details = []
                for metric in low_scores:
                    if current_scores[metric] and current_explanations[metric]:
                        metric_description = {
                            'coherence': 'logical consistency (avoiding contradictions within the assessment)',
                            'completeness': 'comprehensive coverage (including all relevant PHQ-8 symptoms, severities, and duration/frequency details)',
                            'specificity': 'specific details (avoiding vague/generic statements like "the patient seems depressed")',
                            'accuracy': 'accuracy according to DSM-5 or PHQ-8 criteria (correct symptoms and duration/frequency)'
                        }
                        feedback_details.append(f"""
- **{metric.upper()}** scored {current_scores[metric]}/5 because: {current_explanations[metric]}
  Please specifically improve the {metric_description[metric]} in your revised assessment.""")
                
                
                system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible."
                
                feedback_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.

TRANSCRIPT:
{transcript}

ORIGINAL ASSESSMENT:
{qualitative_assessment}

EVALUATION FEEDBACK:
The original assessment has been evaluated and needs improvement in the following areas:
{chr(10).join(feedback_details)}

TASK:
Please reassess the transcript and provide an improved qualitative assessment that addresses the specific feedback above. Build upon the original assessment but focus on improving the identified weaknesses.

Please:
      1. Provide an overall qualitative assessment of the subject's mental health.
      2. Summarize PHQ-8 symptoms if available (if not, state 'not available'), as well as frequency and severity if available.
      3. Summarize social aspects that may influence the subject's mental health. (ex. familial relationships, frienship dynamics, work environment, etc. that are relevant to the subjects mental health)
      4. Summarize biological aspects that may influence the subject's mental health. (ex. famillial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc. that are relevant to the subjects mental health)
      5. Identify potential risk factors the subject may be experiencing.
      6. Use exact quotes from the transcript to support your assessment for each tag.

      Output should be formatted as bullet points with headings for each section using stars. Example: **Tiredness** <explanation of tiredness>. Do not include any additional text outside the bullet points
      Please answer in this XML format with each tag on a new line, properly indented. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:

      <assessment>
        <!-- Summary of participant's overall mental health -->
       <exact_quotes>
        <!-- Quotes from the transcript that support the assessment -->
        </exact_quotes>
      </assessment>

      <PHQ8_symptoms>
        <!-- Summary of PHQ-8 symptoms mentioned in the trancript:
        - Little interest or pleasure in doing things
        - Feeling down, depressed, or hopeless
        - Trouble falling or staying asleep, or sleeping too much
        - Feeling tired or having little energy
        - Poor appetite or overeating
        - Feeling bad about yourself — or that you are a failure or have let yourself or your
          family down
        - Trouble concentrating on things, such as reading the newspaper or watching television
        - Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual
        For each symptom present, note:
        - Frequency (daily, several days, more than half the days, nearly every day)
        - Duration (how long experienced)
        - Severity/impact on functioning

       If symptoms are not discussed, state "not assessed in interview" -->

       <exact_quotes>
        <!-- Quotes from the transcript that support the assessment -->
        </exact_quotes>
      </PHQ8_symptoms>

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
                
                feedback_request = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": feedback_prompt}],
                    "stream": False,
                    "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                }
                
                feedback_response = requests.post(BASE_URL, json=feedback_request, timeout=timeout)
                if feedback_response.status_code != 200:
                    print(f"    Failed to regenerate assessment: {feedback_response.status_code}")
                    failed_evaluations.append(participant_id)
                    break
                
                current_assessment = feedback_response.json()['message']['content']
                print(f"    New assessment generated, re-evaluating only low scores: {low_scores}")
                
                # UPDATED: Only re-evaluate the metrics that scored low
                new_scores = {}
                new_explanations = {}
                
                for metric in low_scores:
                    time.sleep(2)  # Rate limiting
                    
                    if metric == 'coherence':
                        new_coherence_prompt = coherence_prompt.replace(qualitative_assessment, current_assessment)
                        new_coherence_request = {
                            "model": model,
                            "messages": [{"role": "user", "content": new_coherence_prompt}],
                            "stream": False,
                            "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                        }
                        response = requests.post(BASE_URL, json=new_coherence_request, timeout=timeout-10)
                        if response.status_code == 200:
                            content = response.json()['message']['content']
                            score, _ = parse_score_and_explanation(content)
                            new_scores['coherence'] = score
                            new_explanations['coherence'] = content
                            print(f"      Coherence re-evaluated: {score}")
                    
                    elif metric == 'completeness':
                        new_completeness_prompt = completeness_prompt.replace(qualitative_assessment, current_assessment)
                        new_completeness_request = {
                            "model": model,
                            "messages": [{"role": "user", "content": new_completeness_prompt}],
                            "stream": False,
                            "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                        }
                        response = requests.post(BASE_URL, json=new_completeness_request, timeout=timeout-10)
                        if response.status_code == 200:
                            content = response.json()['message']['content']
                            score, _ = parse_score_and_explanation(content)
                            new_scores['completeness'] = score
                            new_explanations['completeness'] = content
                            print(f"      Completeness re-evaluated: {score}")
                    
                    elif metric == 'specificity':
                        new_specificity_prompt = specificity_prompt.replace(qualitative_assessment, current_assessment)
                        new_specificity_request = {
                            "model": model,
                            "messages": [{"role": "user", "content": new_specificity_prompt}],
                            "stream": False,
                            "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                        }
                        response = requests.post(BASE_URL, json=new_specificity_request, timeout=timeout-10)
                        if response.status_code == 200:
                            content = response.json()['message']['content']
                            score, _ = parse_score_and_explanation(content)
                            new_scores['specificity'] = score
                            new_explanations['specificity'] = content
                            print(f"      Specificity re-evaluated: {score}")
                    
                    elif metric == 'accuracy':
                        new_accuracy_prompt = accuracy_prompt.replace(qualitative_assessment, current_assessment)
                        new_accuracy_request = {
                            "model": model,
                            "messages": [{"role": "user", "content": new_accuracy_prompt}],
                            "stream": False,
                            "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
                        }
                        response = requests.post(BASE_URL, json=new_accuracy_request, timeout=timeout-10)
                        if response.status_code == 200:
                            content = response.json()['message']['content']
                            score, _ = parse_score_and_explanation(content)
                            new_scores['accuracy'] = score
                            new_explanations['accuracy'] = content
                            print(f"      Accuracy re-evaluated: {score}")
                
                # Update ONLY the re-evaluated scores, keep the good ones unchanged
                for metric, score in new_scores.items():
                    if score is not None:
                        current_scores[metric] = score
                        current_explanations[metric] = new_explanations[metric]
                
                # Check which scores are STILL low (check all metrics, not just re-evaluated ones)
                low_scores = []
                for metric in ['coherence', 'completeness', 'specificity', 'accuracy']:
                    if current_scores.get(metric) and current_scores[metric] <= 2:
                        low_scores.append(metric)
                
                # Print current scores with indicators for what was re-evaluated
                re_eval_indicators = {metric: " (re-evaluated)" if metric in new_scores else "" 
                                    for metric in ['coherence', 'completeness', 'specificity', 'accuracy']}
                
                print(f"    Iteration {iteration} scores: " + 
                      f"Coherence={current_scores.get('coherence', 'N/A')}{re_eval_indicators['coherence']}, " +
                      f"Completeness={current_scores.get('completeness', 'N/A')}{re_eval_indicators['completeness']}, " +
                      f"Specificity={current_scores.get('specificity', 'N/A')}{re_eval_indicators['specificity']}, " +
                      f"Accuracy={current_scores.get('accuracy', 'N/A')}{re_eval_indicators['accuracy']}")
                
                if low_scores:
                    print(f"    Still have low scores: {low_scores}, continuing with targeted feedback...")
                else:
                    print(f"    All scores now 4 or 5! Enhanced feedback loop complete after {iteration} iterations.")
            
            # Save final results after feedback loop completes
            if iteration >= max_iterations:
                print(f"    Reached max iterations ({max_iterations}), stopping feedback loop")
            
            # Save the final qualitative assessment - keeping original structure
            feedback_assessment_record = {
                'participant_id': participant_id,
                'dataset_type': 'feedback_enhanced',  # Added dataset_type field
                'qualitative_assessment': current_assessment  # Using the improved assessment
            }
            feedback_assessments.append(feedback_assessment_record)
            
            # Save the final evaluation scores - using simplified structure
            feedback_eval_record = {
                'participant_id': participant_id,
                'coherence': current_scores.get('coherence'),
                'coherence_explanation': current_explanations.get('coherence'),
                'completeness': current_scores.get('completeness'),
                'completeness_explanation': current_explanations.get('completeness'),
                'specificity': current_scores.get('specificity'),
                'specificity_explanation': current_explanations.get('specificity'),
                'accuracy': current_scores.get('accuracy'),
                'accuracy_explanation': current_explanations.get('accuracy')
            }
            
            feedback_evaluations.append(feedback_eval_record)
            processed_count += 1
        else:
            print(f"  No low scores detected - skipping feedback loop")
        
        elapsed_time = time.time() - start_time
        print(f"Completed participant {participant_id} in {elapsed_time:.1f}s ({processed_count} with feedback applied)")
            
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        failed_evaluations.append(participant_id)
    
    # Save progress every 10 participants
    if (len(feedback_assessments) % 10 == 0 and len(feedback_assessments) > 0) or len(feedback_assessments) == 1:
        # Save feedback assessments
        if feedback_assessments:
            feedback_assessments_df = pd.DataFrame(feedback_assessments)
            feedback_assessments_df.to_csv(feedback_assessments_csv, index=False)
            print(f"Saved feedback assessments: {len(feedback_assessments)} records to {feedback_assessments_csv}")
        
        # Save feedback evaluations
        if feedback_evaluations:
            feedback_evaluations_df = pd.DataFrame(feedback_evaluations)
            feedback_evaluations_df.to_csv(feedback_evaluations_csv, index=False)
            print(f"Saved feedback evaluations: {len(feedback_evaluations)} records to {feedback_evaluations_csv}")
    
    time.sleep(1)

# Final summary
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total subjects processed: {len(df)}")
print(f"Skipped (no transcript): {skipped_count}")
print(f"Feedback loops applied: {feedback_count}")
print(f"Successfully processed with feedback: {processed_count}")
print(f"Failed: {len(failed_evaluations)}")
print(f"Feedback assessments created: {len(feedback_assessments)}")
print(f"Feedback evaluations created: {len(feedback_evaluations)}")

if failed_evaluations:
    print(f"Failed participant IDs: {failed_evaluations}")

# Save final feedback files
if feedback_assessments:
    feedback_assessments_df = pd.DataFrame(feedback_assessments)
    feedback_assessments_df.to_csv(feedback_assessments_csv, index=False)
    print(f"Final feedback assessments saved: {feedback_assessments_csv}")
    print(f"Feedback assessments CSV columns:")
    print(f"- participant_id")
    print(f"- dataset_type") 
    print(f"- qualitative_assessment")

if feedback_evaluations:
    feedback_evaluations_df = pd.DataFrame(feedback_evaluations)
    feedback_evaluations_df.to_csv(feedback_evaluations_csv, index=False)
    print(f"Final feedback evaluations saved: {feedback_evaluations_csv}")
    print(f"Feedback evaluations CSV columns:")
    print(f"- participant_id")
    print(f"- coherence, coherence_explanation")
    print(f"- completeness, completeness_explanation") 
    print(f"- specificity, specificity_explanation")
    print(f"- accuracy, accuracy_explanation")

if not feedback_assessments and not feedback_evaluations:
    print("No participants required feedback processing, no files created.")