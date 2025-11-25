import json
import requests
import pandas as pd
import os
import time

OLLAMA_NODE = "arctrddgxa002" # TODO: Change this variable to the node where Ollama is running
BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"
model = "gemma3:27b" # TODO: Change this variable to the model you want to use

train_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/train_split_Depression_AVEC2017.csv")
dev_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/dev_split_Depression_AVEC2017.csv")

test_path = pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/test_split_Depression_AVEC2017.csv")

#id_train = train_path.iloc[:, 0].tolist()
#id_dev = dev_path.iloc[:, 0].tolist()

id_test = test_path.iloc[:, 0].tolist()

print(f"Number of train subjects: {len(id_train)}")
print(f"Number of dev subjects: {len(id_dev)}")
print("First 3 train subjects:", id_train[:3] if len(id_train) >= 3 else id_train)
print("First 3 dev subjects:", id_dev[:3] if len(id_dev) >= 3 else id_dev)

#all_subjects = [(subj, 'train') for subj in id_train] + [(subj, 'dev') for subj in id_dev]


all_subjects = [(subj, 'test') for subj in id_test]
print(f"Total subjects to process: {len(all_subjects)}")

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
        
        # prompt
        system_prompt = "You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities. Use exact quotes from the transcript to support your assessment for each tag."
        qual_assessment = f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
        {full_dialogue}
      
       Please:
      1. Provide an overall qualitative assessment of the subject's mental health.
      2. Summarize PHQ-8 symptoms if available (if not, state 'not available'), as well as frequency and severity if available.
      3. Summarize social aspects that may influence the subject's mental health. (for example, familial relationships, frienship dynamics, work environment, etc. that are relevant to the subjects mental health)
      4. Summarize biological aspects that may influence the subject's mental health. (for example, famillial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc. that are relevant to the subjects mental health)
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
        - Feeling bad about yourself — or that you are a failure or have let yourself or your family down
        - Trouble concentrating on things, such as reading the newspaper or watching television
        - Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual
         
         For each symptom present, note:
        - Frequency (not at all, several days, more than half the days, nearly every day)
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
        <!-- Quotes from the transcript that support the assessment -->
        </exact_quotes>
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
        
        print(f"Sending API request...")
        
        # Start timing
        start_time = time.time()
        
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
          }
        )
        
        # End timing
        end_time = time.time()
        runtime_seconds = end_time - start_time
        
        if response.status_code == 200:
            qual_content = response.json()['message']['content']
            print(f"API response received (length: {len(qual_content)} chars)")
            print(f"Response preview: {qual_content[:100]}...")
            print(f"Runtime: {runtime_seconds:.2f} seconds")
            
            # Main results (separated from timing data)
            results.append({
                "participant_id": participant_id,
                "dataset_type": dataset_type,
                "qualitative_assessment": qual_content
            })
            
            runtime_results.append({
                "participant_id": participant_id,
                "status": "success",
                "runtime_seconds": time.time() - start_time
            })
            
            processed_count += 1
            print(f"Completed participant {participant_id} ({processed_count} total completed)")
            
            
            if len(results) == 1 or len(results) % 10 == 0 or len(results) == len(all_subjects):
                # Save main results
                resultsdf = pd.DataFrame(results)
                output_file = "/data/users2/user/new_analysis_results/TESTSUBJECTS.csv"
                resultsdf.to_csv(output_file, index=False)
                print(f"Checkpoint save: {len(results)} participants saved to {output_file}")
                
                # Save timing results
                timing_df = pd.DataFrame(timing_results)
                timing_file = "/data/users2/user/new_analysis_results/qual_runtime_GEMMA.csv"
                timing_df.to_csv(timing_file, index=False)
                print(f"Timing checkpoint save: {len(timing_results)} participants saved to {timing_file}")
                
                
                print(f"\n=== FORMATTED PREVIEW OF PARTICIPANT {participant_id} ===")
                print(qual_content)
                print("=" * 60)
            
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            runtime_results.append({
                "participant_id": participant_id,
                "status": "failed",
                "runtime_seconds": time.time() - start_time
            })
            
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
