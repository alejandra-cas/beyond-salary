

import json
import os
import pandas as pd
import time
import tiktoken
import requests
from openai import OpenAI
from openai import OpenAIError
import pickle
import re

os.environ['OPENAI_API_KEY'] = ##
client = OpenAI()

# Rate limit parameters
max_requests_per_minute = 3500
max_tokens_per_minute = 60000
max_requests_per_day = 10000

# Time tracking
start_time = time.time()
request_count = 0
token_count = 0
total_requests_today = 0

input_path = '../Data/test_for_gpt_classification.csv'
checkpoint_path = '../Exports/GPT Prompt Tests/job_flexibility_results_checkpoint_prompt11_survey.csv'
pickle_path = '../Exports/GPT Prompt Tests/job_flexibility_latest_batch.pkl'

responses = []
failed_responses = []

def clean_description(description):
    # Remove unwanted characters or formatting
    cleaned_description = description.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned_description = ' '.join(cleaned_description.split())  # Remove extra spaces
    return cleaned_description

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)

def classify_job_descriptions(job_ids, job_descriptions):
    job_descriptions = [clean_description(desc) for desc in job_descriptions]
    model = "gpt-3.5-turbo"
    
    prompts = [f"Job ID: {job_id}\n<Body>: {desc}" for job_id, desc in zip(job_ids, job_descriptions)]
    
    messages = [
        {
            "role": "system",
            "content": f"""You are an expert in evaluating job descriptions for flexibility. Return the results in JSON."""
        },
        {
            "role": "user",
            "content": f""" 
        For each Job ID, using only the information provided under <Body>, evaluate whether it offers the following dimensions of flexibility using 'True', 'False', or 'Unknown'. Only answer 'True' if it is explicitly stated in the body text. Also, rate how flexible the job is overall from 1 to 5, where 5 is most flexible.
        1. Remote: Offers the option work remotely. e.g. work from home, remote work, work from anywhere
        2. Leave Flexibility: Can take time during the work day to attend to personal/family responsibilities
        3. Schedule Flexibility: Flexibility to choose working hours. Does not include employee willingness to work abnormal hours or requirements to be available at given times.
        4. Overall Flexibility Rating: Can only be a number from 1 to 5
        
        Here is an example:
        Input:
        Job ID: 12345
        <Body>
        Join us as a Data Scientist on our AI team. Our team is distributed and you have the freedom to choose when and where you work. Benefits: - flexible work schedule - 401k match.
        
        Job ID: 67890
        <Body>
        We are looking for an AI specialist. We value work-life balance and support flexible arrangements so you can take care of the demands life brings. We are currently working from the office 20% of the time.  You should be able to work during Eastern Time Zone working hours.
        
        Job ID: 89102
        <Body>
        Title: Analytics Engineer. At our company, we work hard to get the best results. Willingness to work flexible hours including evenings, weekends, and holidays as needed. Must be able to commute to one of our offices. 
        
        
        Output:
        [
            {{
                "Job ID": "12345",
                "Remote": "True",
                "Leave Flexibility": "True",
                "Schedule Flexibility": "True",
                "Overall Flexibility Rating": "5"
            }},
            {{
                "Job ID": "67890",
                "Remote": "True",
                "Leave Flexibility": "True",
                "Schedule Flexibility": "False",
                "Overall Flexibility Rating": "3"
            }},
            {{
                "Job ID": "89102",
                "Remote": "False",
                "Leave Flexibility": "False",
                "Schedule Flexibility": "False",
                "Overall Flexibility Rating": "1"
            }}                              
        ]
        
        Evaluate the following jobs as instructed:
        """
        }
    ]
    
    # Create the final message with all job descriptions batched together
    messages[1]["content"] += "\n\n".join(prompts)
    
    try:
        api_start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        api_time = time.time() - api_start
        print(f"API time: {api_time}")
        append_start = time.time()
        responses.append(response)
        append_time = time.time() - append_start
        print(f"Append time: {append_time}")
        token_usage = response.usage.total_tokens
        # usage_list.append(token_usage)
        with open('../exports/gpt_responses.pkl', 'wb') as f:
            pickle.dump(responses, f)       
        return response.choices[0].message.content, token_usage
    except OpenAIError as e:
        print(f"An error occurred: {e}")
        #save responses to pickle
        with open('../exports/gpt_responses.pkl', 'wb') as f:
            pickle.dump(responses, f)
        return None, None


def normalize_response(response_content):
    try:
        # Print raw response for debugging
        # print(f"Raw response content: {response_content}")

        # Remove the triple backticks and `json` label
        if response_content.startswith("```json\n"):
            response_content = response_content[len("```json\n"):]
        if response_content.endswith("\n```"):
            response_content = response_content[:-len("\n```")]
            
        start_index = response_content.find('[')
        end_index = response_content.rfind(']') + 1
        if start_index == -1 or end_index == -1:
            raise ValueError("The response content is not a valid JSON array.")
        
        response_content = response_content[start_index:end_index]
        # Replace non-standard boolean and unknown values
        normalized_content = response_content.replace('"True"', 'true').replace('"False"', 'false')
        # Fix the Overall Flexibility Rating values, handle both quoted and unquoted numbers
       
        normalized_content = re.sub(r'"Overall Flexibility Rating":\s*"(\d+)"', r'"Overall Flexibility Rating": \1', normalized_content)
        normalized_content = re.sub(r'"Overall Flexibility Rating":\s*(\d+)"', r'"Overall Flexibility Rating": \1', normalized_content)

        # Load JSON to ensure it's properly formatted
        response_data = json.loads(normalized_content)
                # Ensure response_data is a list of dictionaries
        if isinstance(response_data, dict):
            response_list = [response_data]
        elif isinstance(response_data, list):
            response_list = response_data
        else:
            failed_responses.append(response_content)
            raise ValueError("Unexpected response format")
        
        return response_list
    except json.JSONDecodeError as e:
        failed_responses.append(response_content)
        with open('../exports/gpt_responses.pkl', 'wb') as f:
            pickle.dump(responses, f)
        print(f"JSON decode error: {e}")
        print(f"Raw response content: {response_content}")
        return None
    
def process_requests(df):
    global request_count, token_count, total_requests_today, start_time

    max_tokens_per_request = 4097
    tokens_per_prompt = 508
    tokens_per_output = 55

    results = []
    new_results = []
    checkpoint_file = checkpoint_path
    pickle_file = pickle_path
    
    if os.path.exists(checkpoint_file):
        existing_results_df = pd.read_csv(checkpoint_file)
        processed_job_ids = set(existing_results_df['Job ID'])
    else:
        existing_results_df = pd.DataFrame()
        processed_job_ids = set()

    i = 0
    while i < len(df):
        batch_df = []
        current_tokens = 6 + tokens_per_prompt
        
        start_batching = time.time()
        while i < len(df) and current_tokens < max_tokens_per_request:
            job_id = df.iloc[i]['ID']
            job_description = df.iloc[i]['BODY']
            
            if job_id in processed_job_ids:
                i += 1
                continue
            
            job_tokens = count_tokens(job_description) + tokens_per_output
            
            if current_tokens + job_tokens > max_tokens_per_request:
                break
            
            batch_df.append((job_id, job_description))
            current_tokens += job_tokens
            i += 1
            if i%10 ==0:
                print(f"Processing job description {i}")
        
        if not batch_df:
            i += 1
            continue
        batching_duration = time.time() - start_batching
        print(f"Batching duration: {batching_duration:.2f} seconds")
        # job_ids = batch_df['ID'].tolist()
        # job_descriptions = batch_df['BODY'].tolist()
        job_ids, job_descriptions = zip(*batch_df)
        # print(job_ids)
        # print(job_descriptions)
        # Throttling to stay within rate limits
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            if request_count >= max_requests_per_minute or token_count + current_tokens >= max_tokens_per_minute:
                print(f"Throttling: sleeping for {60 - elapsed_time:.2f} seconds")
                time.sleep(60 - elapsed_time)
                start_time = time.time()
                request_count = 0
                token_count = 0
                total_requests_today += request_count

        if total_requests_today >= max_requests_per_day:
            print("Daily request limit reached.")
            break
        
        start_classification = time.time()
        result, token_usage = classify_job_descriptions(job_ids, job_descriptions)
        print(f"Classifying job descriptions through {i}")
        classification_duration = time.time() - start_classification
        print(f"Classification duration for batch: {classification_duration:.2f} seconds")  # Print classification duration       
        
        request_count += 1
        token_count += token_usage

        if result:
            # print(result)
            start_normalization = time.time()  # Start timing the normalization            
            normalized_result = normalize_response(result)
            normalization_duration = time.time() - start_normalization  # End timing the normalization
            print(f"Normalization duration: {normalization_duration:.2f} seconds")  # Print normalization duration
            add_results = time.time()
            if normalized_result:
                for result_dict in normalized_result:
                    job_id = result_dict["Job ID"]
                    if job_id not in processed_job_ids:
                        new_results.append(result_dict)
                        processed_job_ids.add(job_id)
                        
                        new_result_df = pd.DataFrame([result_dict])
                        existing_results_df = pd.concat([existing_results_df, new_result_df], ignore_index=True)
                        existing_results_df.to_csv(checkpoint_file, index=False)
            else:
                print(f"Failed to normalize response for Job ID batch {job_ids}")
                print(result)
                return result
            add_results_duration = time.time() - add_results
            print(f"Adding results duration: {add_results_duration:.2f} seconds")
    results_df = pd.DataFrame(new_results)
    # print(f"Saving results to pickle file: {time.time()}")
    # results_df.to_pickle(pickle_file)
    # print(f"Results saved to {pickle_file}: {time.time()}")
    return results_df

df = pd.read_csv(input_path)
process_requests(df)