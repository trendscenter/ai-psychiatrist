import json
import requests
import pandas as pd
import os
import json
import requests
import pandas as pd
import os
import time

OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3:27b" # TODO: Change this variable to the model you want to use

train_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
dev_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")

test_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/test_split_Depression_AVEC2017.csv")

#id_train = train_path.iloc[:, 0].tolist()
#id_dev = dev_path.iloc[:, 0].tolist()

id_test = test_path.iloc[:, 0].tolist()

#all_subjects = [(subj, 'train') for subj in id_train] + [(subj, 'dev') for subj in id_dev]

all_subjects = [(subj, 'test') for subj in id_test]
print(f"Total subjects to process: {len(all_subjects)}")

# PHQ-8 symptoms list
phq8_symptoms = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"
]

results = []
runtime_results = []
processed_count = 0
skipped_count = 0

for i, (participant_id, dataset_type) in enumerate(all_subjects):
    print(f"\n--- Processing {i+1}/{len(all_subjects)}: {participant_id} ({dataset_type} dataset) ---")
    
    start_time = time.time()
    
    id_transcript = os.path.join("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/", f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")
    
    print(f"Looking for transcript at: {id_transcript}")
    
    if not os.path.exists(id_transcript):
        print(f"Transcript not found for {participant_id}")
        skipped_count += 1
        runtime_results.append({
            "participant_id": participant_id,
            "status": "skipped",
            "runtime_seconds": time.time() - start_time
        })
        continue

    print(f"Transcript found, loading data...")
    
    try:
        transcript_data = pd.read_csv(id_transcript, sep='\t')
        dialogue_txt = transcript_data.dropna(subset=['speaker', 'value'])
        dialogue_txt['dialogue'] = dialogue_txt['speaker'] + " : " + dialogue_txt['value']
        full_dialogue = "\n".join(dialogue_txt['dialogue'])
        
        print(f"Dialogue length: {len(full_dialogue)} characters")
        print(f"First 200 chars: {full_dialogue[:200]}...")
        
        # Overall assessment
        system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities."
        
        print(f"Sending API request for overall assessment...")
        overall_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Provide an overall qualitative assessment of the subject's mental health.

Please answer in this XML format. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:

<assessment>Summary of participant's overall mental health</assessment>
<quotes>Exact quotes from the transcript that support the assessment</quotes>
"""
        
        # Start timing
        start_time = time.time()
        
        response = requests.post(
            BASE_URL,
            json = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": overall_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            runtime_results.append({
                "participant_id": participant_id,
                "status": "failed",
                "runtime_seconds": time.time() - start_time
            })
            continue
            
        overall_content = response.json()['message']['content']
        
        # Check tags
        if "<assessment>" not in overall_content or "<quotes>" not in overall_content:
            print("WARNING: Missing tags in overall assessment")
        
        # PHQ-8 symptoms - assess each one
        phq8_results = []
        for symptom in phq8_symptoms:
            print(f"Assessing PHQ-8 symptom: {symptom[:50]}...")
            
            symptom_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Assess this specific PHQ-8 symptom: {symptom}

For this symptom, note:
- Frequency (daily, several days, more than half the days, nearly every day)
- Duration (how long experienced)
- Severity/impact on functioning
- If not discussed, state "not assessed in interview"

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of this symptom's presence, frequency, duration, and severity</assessment>
<quotes>Exact quotes from the transcript that support this assessment</quotes>
"""
            
            response = requests.post(
                BASE_URL,
                json = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": symptom_prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_k": 20,
                        "top_p": 0.9
                    }
                }
            )
            
            if response.status_code != 200:
                phq8_results.append(f"ERROR: Failed to assess symptom")
                continue
                
            symptom_content = response.json()['message']['content']
            
            if "<assessment>" not in symptom_content or "<quotes>" not in symptom_content:
                print(f"WARNING: Missing tags for symptom assessment")
            
            phq8_results.append(f"**{symptom}**\n{symptom_content}")
        
        # Social factors
        print(f"Assessing social factors...")
        social_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Summarize social aspects that may influence the subject's mental health (ex. familial relationships, friendship dynamics, work environment, etc.).

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of social influences on patient's health</assessment>
<quotes>Quotes from the transcript that support the assessment</quotes>
"""
        
        response = requests.post(
            BASE_URL,
            json = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": social_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code != 200:
            print(f"API request failed for social factors")
            social_content = "ERROR"
        else:
            social_content = response.json()['message']['content']
            if "<assessment>" not in social_content or "<quotes>" not in social_content:
                print(f"WARNING: Missing tags for social factors")
        
        # Biological factors
        print(f"Assessing biological factors...")
        biological_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Summarize biological aspects that may influence the subject's mental health (ex. familial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc.).

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of biological influences on patient's health</assessment>
<quotes>Quotes from the transcript that support the assessment</quotes>
"""
        
        response = requests.post(
            BASE_URL,
            json = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": biological_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code != 200:
            print(f"API request failed for biological factors")
            biological_content = "ERROR"
        else:
            biological_content = response.json()['message']['content']
            if "<assessment>" not in biological_content or "<quotes>" not in biological_content:
                print(f"WARNING: Missing tags for biological factors")
        
        # Risk factors
        print(f"Assessing risk factors...")
        risk_prompt = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{full_dialogue}

Identify potential risk factors the subject may be experiencing.

Please answer in this XML format. Use straight quotes instead of curly quotes:

<assessment>Summary of potential risk factors</assessment>
<quotes>Exact quotes from the transcript that support the assessment</quotes>
"""
        
        response = requests.post(
            BASE_URL,
            json = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": risk_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code != 200:
            print(f"API request failed for risk factors")
            risk_content = "ERROR"
        else:
            risk_content = response.json()['message']['content']
            if "<assessment>" not in risk_content or "<quotes>" not in risk_content:
                print(f"WARNING: Missing tags for risk factors")
        
        # Combine all results
        final_assessment = f"""=== OVERALL ASSESSMENT ===
{overall_content}

=== PHQ-8 SYMPTOMS ===
{chr(10).join(phq8_results)}

=== SOCIAL FACTORS ===
{social_content}

=== BIOLOGICAL FACTORS ===
{biological_content}

=== RISK FACTORS ===
{risk_content}
"""
        
        runtime_seconds = time.time() - start_time
        print(f"Runtime: {runtime_seconds:.2f} seconds")
        
        results.append({
            "participant_id": participant_id,
            "dataset_type": dataset_type,
            "qualitative_assessment": final_assessment
        })
        
        runtime_results.append({
            "participant_id": participant_id,
            "status": "success",
            "runtime_seconds": runtime_seconds
        })
        
        processed_count += 1
        print(f"Completed participant {participant_id} ({processed_count} total completed)")
        
        if len(results) == 1 or len(results) % 10 == 0 or len(results) == len(all_subjects):
            resultsdf = pd.DataFrame(results)
            output_file = "/home/users/nblair7/ai-psychiatrist/qualitative_assessment/MG142ipy2.csv"
            resultsdf.to_csv(output_file, index=False)
            print(f"Checkpoint save: {len(results)} participants saved to {output_file}")
            
            timing_df = pd.DataFrame(runtime_results)
            timing_file = "/home/users/nblair7/ai-psychiatrist/qualitative_assessment/MG142ipyrun2.csv"
            timing_df.to_csv(timing_file, index=False)
            print(f"Timing checkpoint save: {len(runtime_results)} participants saved to {timing_file}")
            
            print(f"\n=== FORMATTED PREVIEW OF PARTICIPANT {participant_id} ===")
            print(final_assessment)
            print("=" * 60)
            
    except Exception as e:
        print(f"Error processing {participant_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        runtime_results.append({
            "participant_id": participant_id,
            "status": "error",
            "runtime_seconds": time.time() - start_time
        })
        continue

print(f"Summary of processing:")
print(f"Total subjects: {len(all_subjects)}")
print(f"Successfully processed: {processed_count}")
print(f"Skipped (no transcript): {skipped_count}")
print(f"Results collected: {len(results)}")

if results:
    # Save main results file
    resultsdf = pd.DataFrame(results)
    output_file = "/data/users2/user/new_analysis_results/TESTSUBJECTS.csv"
    resultsdf.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")

if runtime_results:
    runtime_df = pd.DataFrame(runtime_results)
    runtime_output_file = "/home/users/user/ai-psychiatrist/runtime_results.csv"
    runtime_df.to_csv(runtime_output_file, index=False)
    print(f"Saved runtime results to: {runtime_output_file}")
else:
    print("No results to save!")