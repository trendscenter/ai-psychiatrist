import os
import csv
import json
import requests
import pandas as pd

LLAMA_API_KEY = ""
BASE_URL = "https://api.llama.com/v1/chat/completions" # llama4 server URL
# BASE_URL = "https://arctrddgxa003:11434/api/chat" # ollama server URL

def load_transcript(data_path, id):
    transcript_path = os.path.join(data_path, f"{id}_P/{id}_TRANSCRIPT.csv")
    df = pd.read_csv(transcript_path)
    with open(transcript_path, 'r') as file:
        header = file.readline().split("\t")
        interview = ""
        for line in file:
            parts = line.split("\t")
            interview += parts[2]+": "+parts[3]
    return interview

def chat_completion(messages, model="Llama-4-Maverick-17B-128E-Instruct-FP8", max_tokens=2048):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}"
    }

    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_k": 10,
            "top_p": 0.1
        }
    }
    
    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload
    )

    return response.json()

def extract_response(response_text, key_list):
    output_dict = {}
    for k in key_list:
        try:
            idx1 = response_text.index(f"<{k}>")
            idx2 = response_text.index(f"</{k}>")
            res = response_text[idx1 + len(f"<{k}>") : idx2]
            if "score" in k:
                    output_dict[k] = int(res)
            else:
                output_dict[k] = res
        except ValueError:
                output_dict[k] = "N/A"
    return output_dict

def main():
    data_path = "/data/users4/xli/ai-psychiatrist/datasets/daic_woz_dataset"
    phq8_csv_path = os.path.join(data_path, "train_split_Depression_AVEC2017.csv")
    df_phq8 = pd.read_csv(phq8_csv_path)

    id_list = list(df_phq8["Participant_ID"])
    output_list = []
    output_filename = 'test_assessment.csv'

    for id in id_list[:2]:
        print(f"participant {id}")
        interview = load_transcript(data_path, id)

        system_prompt = "You are an AI psychiatrist assistant specializing in depression. Your task is to analyze interview transcripts and assess the patient's mental health status, focusing on signs of depression."

        qualitative_assessment_prompt = f"""Here is the interview transcript:
        {interview}

        Please analyze the interview transcript above and assess the patient's mental health status, focusing on signs of depression, from social and biological aspects. Please think step-by-step and provide your assessment in a structured format, including the following sections:
        1. Overall Impression in <overall> tags: A brief summary of the patient's mental health status.
        2. Social Aspects in <social> tags: Observations related to the patient's interpersonal relationships, family situation, and any relevant social interactions that may influence their mental health. Find relevant quotes from the interview transcript.
        3. Biological Aspects in <biological> tags: Observations related to the patient's physical health, including sleep quality, appetite, physical activity, stress level, and any other biological factors that may influence their mental health. Find relevant quotes from the interview transcript.
        4. Additional Notes in <notes> tags: Any other relevant observations or comments that do not fit into the above categories.
        5. Potential Risk Factors in <risks> tags: Any identified risk factors for depression or other mental health issues."""

        qualitative_assessment_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qualitative_assessment_prompt}
        ]

        qualitative_assessment_response = chat_completion(qualitative_assessment_messages, model="gemma3-optimized:27b")
        try:
            qualitative_assessment_response_text = qualitative_assessment_response["completion_message"]["content"]["text"]
            # print(qualitative_assessment_response_text)
        except KeyError:
            print("qualitative assessment error")
            continue

        quantitative_assessment_prompt = f"""Here is the interview transcript:
        {interview}

        Please analyze the interview transcript above, assess the patient's signs of depression, and predict the eight-item Patient Health Questionnaire depression scale (PHQ-8) scores. The PHQ-8 counts the number of days in the past 2 weeks the respondent had experienced a particular depressive symptom, and assigns points (0 to 3) to each category: (1) 0 point: 0 to 1 day ("not at all"); (2) 1 point: 2 to 6 days ("several days"); (3) 2 points: 7 to 11 days ("more than half the days"); (4) 3 points: 12 to 14 days ("nearly every day"). Please think step-by-step and provide your assessment in a structured format, including the following assessments:
        1. When the patient reported little interest or pleasure in doing things, find relevant quotes in <nointerest_quotes> tags, assess relevant symptoms in <nointerest_assessment> tags, and only provide an integer score in <nointerest_score> tags.
        2. When the patient reported feeling down, depressed, or hopeless, find relevant quotes in <depressed_quotes> tags, assess relevant symptoms in <depressed_assessment> tags, and only provide an integer score in <depressed_score> tags.
        3. When the patient reported trouble falling or staying asleep, or sleeping too much, find relevant quotes in <sleep_quotes> tags, assess relevant symptoms in <sleep_assessment> tags, and only provide an integer score in <sleep_score> tags.
        4. When the patient reported feeling tired or having little energy, find relevant quotes in <tired_quotes> tags, assess relevant symptoms in <tired_assessment> tags, and only provide an integer score in <tired_score> tags.
        5. When the patient reported poor appetite or overeating, find relevant quotes in <appetite_quotes> tags, assess relevant symptoms in <appetite_assessment> tags, and only provide an integer score in <appetite_score> tags.
        6. When the patient reported feeling bad about themselves, or that they are a failure or have let themselves or their family down, find relevant quotes in <failure_quotes> tags, assess relevant symptoms in <failure_assessment> tags, and only provide an integer score in <failure_score> tags.
        7. When the patient reported trouble concentrating on things, such as reading the newspaper or watching television, find relevant quotes in <concentrating_quotes> tags, assess relevant symptoms in <concentrating_assessment> tags, and only provide an integer score in <concentrating_score> tags.
        8. When the patient reported moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that they have been moving around a lot more than usual, find relevant quotes in <moving_quotes> tags, assess relevant symptoms in <moving_assessment> tags, and only provide an integer score in <moving_score> tags."""

        quantitative_assessment_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": quantitative_assessment_prompt}
        ]

        quantitative_assessment_response = chat_completion(quantitative_assessment_messages, model="gemma3-optimized:27b")
        try:
            quantitative_assessment_response_text = quantitative_assessment_response["completion_message"]["content"]["text"]
        except KeyError:
            print("quantitative assessment error")
            continue
        # print(quantitative_assessment_response_text)

        qualitative_assessment_key_list = ["overall", "social", "biological", "notes", "risks"]
        qualitative_assessment_dict = extract_response(qualitative_assessment_response_text, qualitative_assessment_key_list)

        quantitative_assessment_key_list = []
        for i in ["nointerest", "depressed", "sleep", "tired", "appetite", "failure", "concentrating", "moving"]:
            for j in ["quotes", "assessment", "score"]:
                quantitative_assessment_key_list.append(f"{i}_{j}")
        quantitative_assessment_dict = extract_response(quantitative_assessment_response_text, quantitative_assessment_key_list)

        output_dict = {'participant_id': id, 'qualitative_assessment': qualitative_assessment_response_text, 'quantitative_assessment': quantitative_assessment_response_text}
        output_dict.update(qualitative_assessment_dict)
        output_dict.update(quantitative_assessment_dict)

        output_list.append(output_dict)

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['participant_id', 'qualitative_assessment', 'quantitative_assessment']
        fieldnames += qualitative_assessment_key_list
        fieldnames += quantitative_assessment_key_list
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_list)

if __name__ == "__main__":
    main()