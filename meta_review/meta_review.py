import os
import csv
import json
import requests
import pandas as pd

def load_jsonl(file_path):
    """
    Loads the jsonl file

    Parameters
    ----------
    file_path : string
        The path to the jsonl file

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents 
        a JSON object from a line in the JSONL file
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_response(response_text, key_list):
    output_dict = {}
    for k in key_list:
        try:
            idx1 = response_text.index(f"<{k}>")
            idx2 = response_text.index(f"</{k}>")
            output_dict[k] = response_text[idx1 + len(f"<{k}>") : idx2]
        except ValueError:
                output_dict[k] = "N/A"
    return output_dict

def main():
    OLLAMA_NODE = "arctrdgn001" # TODO: Change this variable to the node where Ollama is running
    BASE_URL = f"http://{OLLAMA_NODE}:11434/api/chat"

    model = "gemma3-optimized:27b" # TODO: Change this variable to the model you want to use

    rootdir = "/data/users4/user/ai-psychiatrist"

    output_list = []
    output_filename = "meta_review_test.csv"

    # Load qualitative assessment
    qual_df = pd.read_csv(os.path.join(rootdir, "analysis_output/qual/qual_assessment_GEMMA.csv"))

    # Load quantitative assessment
    quan_list = load_jsonl(os.path.join(rootdir, "analysis_output/quan/chunk_8_step_2_examples_2_embedding_results_analysis_2.jsonl"))

    phq8_questions = [
        'PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
        'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving'
    ]

    quan_list_ = []
    for i in quan_list:
        quan_dict = {"participant_id": i["participant_id"]}
        for j in phq8_questions:
            quan_dict[f"{j}"] = i[j]["score"]
            quan_dict[f"{j}_Reason"] = i[j]["reason"]
        quan_list_.append(quan_dict)

    quan_df = pd.DataFrame(quan_list_)
    participant_id_list = quan_df["participant_id"].tolist()
    participant_id_list.sort()

    for participant_id in participant_id_list:
        # Load transcript
        transcript_path = os.path.join(rootdir, f"datasets/daic_woz_dataset/{participant_id}_P/{participant_id}_TRANSCRIPT.csv")
        raw_transcript = pd.read_csv(transcript_path, sep="\t")
        raw_transcript['speaker'] = raw_transcript['speaker'].fillna('Unknown').astype(str)
        raw_transcript['value'] = raw_transcript['value'].fillna('').astype(str)
        transcript = '\n'.join(raw_transcript['speaker'] + ': ' + raw_transcript['value'])

        # print(transcript)

        qual_output = qual_df.loc[qual_df['participant_id'] == participant_id, 'qualitative_assessment'].iloc[0][7:-4]

        # print(qual_output)

        ind = quan_df['participant_id'] == participant_id
        quan_output = ""
        for i in phq8_questions:
            score = quan_df.loc[ind, i].iloc[0]
            key_reason = i+"_Reason"
            reason = quan_df.loc[ind, key_reason].iloc[0]
            if score != "N/A":
                quan_output += f"<{i.lower()}_score>{score}</{i.lower()}_score>\n"
                quan_output += f"<{i.lower()}_explanation>{reason}</{i.lower()}_explanation>\n"

        # print(quan_output)

        system_prompt = "You are an AI psychiatrist assistant specializing in depression. Your task is to analyze the interview transcript and related qualitative and quantitative assessments, and then predict the diagnosis label and severity level."

        meta_reviewer_prompt = f"""Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.  

        Here is the interview transcript in <transcript> tags:
        <transcript>
        {transcript}
        </transcript>

        Here are the overall assessment, social and biological risk factors that may influence the participant's mental health in <qualitative_assessment> tags:
        <qualitative_assessment>
        {qual_output}
        </qualitative_assessment>

        Here are the predicted PHQ-8 scores and explanations based on the available information in the interview transcript in <quantitative_assessment> tags:
        <quantitative_assessment>\n{quan_output}</quantitative_assessment>

        According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; 15 to 19, moderately severe; and 20 to 24, severe. \
        Note that the interview may not cover all eight PHQ-8 items. Therefore, directly summing the available scores and deriving a severity level could underestimate the participant's condition. Please use the available information to infer and predict the participant's condition as accurately as possible. \
        Please predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
        Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
        """

        # print(meta_reviewer_prompt)

        response = requests.post(
        BASE_URL,
        json = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": meta_reviewer_prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_k": 20,
                "top_p": 1
            }
        }
        )

        # print(json.dumps(response.json(), indent=2))

        try:
            response_text = response.json()["message"]["content"]
            print(response_text)
        except KeyError:
            print("error")

        output_key_list = ["severity", "explanation"]
        prediction_dict = extract_response(response_text, output_key_list)

        output_dict = {"participant_id": participant_id, "response": response_text}
        output_dict.update(prediction_dict)

        output_list.append(output_dict)

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ["participant_id", "response", "severity", "explanation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_list)

if __name__ == "__main__":
    main()