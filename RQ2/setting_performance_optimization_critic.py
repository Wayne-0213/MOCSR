import openai
import pandas as pd
import time
import re
import json
from tqdm import tqdm
import sys
sys.path.append("../..")
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FEWSHOT_PATH = ROOT_DIR / 'dataset' / 'human_evaluation' / 'RQ1_human_evaluation_results.xlsx'
INPUT_JSONL_PATH = ROOT_DIR / 'result' / 'EP4CS' / 'Java' / 'test_new_pred_EP4CS.jsonl'
START_INDEX = 0
NUM_SAMPLES = 1000
FOUR_DIM_COLS = ['Coherence', 'Consistency', 'Fluency', 'Relevance']
CRITIC_DIM_COLS = [f'{dim}_critic' for dim in FOUR_DIM_COLS]
CHAT_COMPLETION_MODELS = {'gpt-4.1-nano', 'gpt-4o-mini', 'gpt-4o', 'deepseek-v4-pro-guan'}
FEWSHOT_NUM = NUM_SAMPLES
_EVAL_DATA_OVERRIDE = None

openai.api_key = os.getenv('OPENAI_API_KEY', '')
_base_url = os.getenv('OPENAI_BASE_URL', 'https://www.dmxapi.cn').rstrip('/')
if not _base_url.endswith('/v1'):
    _base_url = _base_url + '/v1'
openai.base_url = _base_url + '/'


def get_reference_summary(item):
    if item.get('ref_summary'):
        return item.get('ref_summary', '')
    if item.get('docstring'):
        return item.get('docstring', '')

    doc_tokens = item.get('docstring_tokens', [])
    if isinstance(doc_tokens, list):
        return ' '.join(str(token) for token in doc_tokens)
    return str(doc_tokens or '')


def get_generated_summary(item):
    return item.get('pred_EP4CS') or item.get('pred_summary') or ''


def is_full_score(value):
    try:
        return float(value) == 4.0
    except (TypeError, ValueError):
        return False


def load_eval_data_from_jsonl(num):
    rows = []
    with INPUT_JSONL_PATH.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if idx < START_INDEX:
                continue
            if idx >= START_INDEX + num:
                break
            item = json.loads(line)
            rows.append(
                {
                    'sample_id': idx,
                    'Code': item.get('code', ''),
                    'Target': get_reference_summary(item),
                    'Generated': get_generated_summary(item),
                    **{dim_name: item.get(dim_name) for dim_name in FOUR_DIM_COLS},
                }
            )
    return pd.DataFrame(rows)


def count_jsonl_rows(path):
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def normalize_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0

    if not 0 <= score <= 4:
        return 0
    if score.is_integer():
        return int(score)
    return score


def write_scores_back_to_jsonl(scores_df):
    missing_cols = [col for col in ['sample_id'] if col not in scores_df.columns]
    if missing_cols:
        raise ValueError(f'Missing score columns: {missing_cols}')

    rows_by_id = {int(row['sample_id']): row for _, row in scores_df.iterrows()}
    tmp_path = INPUT_JSONL_PATH.with_suffix(INPUT_JSONL_PATH.suffix + '.tmp')
    written_ids = set()

    with INPUT_JSONL_PATH.open('r', encoding='utf-8') as fin, tmp_path.open('w', encoding='utf-8') as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue

            obj = json.loads(line)
            row = rows_by_id.get(idx)
            if row is not None:
                for dim_name in FOUR_DIM_COLS:
                    field_name = f'{dim_name}_critic'
                    if is_full_score(obj.get(dim_name)):
                        obj.pop(field_name, None)
                    elif field_name in row and not pd.isna(row[field_name]):
                        obj[field_name] = str(row[field_name])
                written_ids.add(idx)
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    missing_ids = sorted(set(rows_by_id) - written_ids)
    if missing_ids:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f'Score rows could not be written back for sample_id values: {missing_ids[:10]}')

    tmp_path.replace(INPUT_JSONL_PATH)

def evaluate(num, model, reference, nshot):

    df_score = pd.read_excel(FEWSHOT_PATH).iloc[:num]

    coh_example = df_score[df_score['Coherence'].apply(lambda x: x.is_integer())].groupby('Coherence').head(nshot)
    con_example = df_score[df_score['Consistency'].apply(lambda x: x.is_integer())].groupby('Consistency').head(nshot)
    flu_example = df_score[df_score['Fluency'].apply(lambda x: x.is_integer())].groupby('Fluency').head(nshot)
    ref_example = df_score[df_score['Relevance'].apply(lambda x: x.is_integer())].groupby('Relevance').head(nshot)

    criteria = {
        "Coherence": "the summary should be well-structured and well-organized. The summary should not just be a heap "
                     "of related information, but should build from sentence to sentence to a coherent body of "
                     "information about a topic.",

        "Consistency": "the factual alignment between the summary and the summarized code. A factually consistent "
                       "summary contains only statements that are entailed by the source code. Annotators were "
                       "also asked to penalize summaries that contained hallucinated facts. ",

        "Fluency": "the quality of individual sentences. The sentence should have no repetitive word, formatting "
                   "problems, capitalization errors or obviously ungrammatical sentences ( "
                   "e.g., fragments, missing components) that make the text difficult to understand.",

        "Relevance": "selection of important content from the source. The summary should include only important "
                     "information from the source document. Annotators were instructed to penalize summaries that "
                     "contained redundancies and excess information.",
    }

    # example

    if reference:
        roles = {
            # coherence
            "Original Code Author": "As the Original Code Author, having written the code, you ensure the coherence of the "
                                    "code summary, ensuring that it clearly conveys the main logic of the code.",
            # Consistency
            "Code Reviewer1": "As a Code Reviewer, serving as an experienced developer, you guarantee that the summary "
                              "remains consistent with the original code. You ensure that the summary captures the "
                              "primary functionality and logic of the code without introducing any additional or "
                              "unrelated content.",
            # Fluency
            "Code Reviewer2": "As a Code Reviewer, serving as an experienced developer, you focus on ensuring that the summary is written smoothly, with clear "
                              "sentences and appropriate wording. You challenge other judgments and provide alternative "
                              "solutions when necessary.",
            # Relevance
            "Code Editor": "As a Code Editor, concentrating on the business or functional relevance of the code, "
                           "you ensure that the summary captures the key significance of the code in the larger "
                           "system or project.",
        }
        evaluation_step = {
            'Coh': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its main functionality and key operations.'
            # '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
            # 'the main functionality and key operations of the code, and if they present them in a clear and '
            # 'logical order. '
            # '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Con': '',
            # 'Evaluation Steps:'
            # '1. Read the Source Code carefully and understand its main functionality and any key operations.'
            # '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
            # 'Ensure that the summary contains only statements that are present or implied in the source code.'
            # 'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
            # 'source code. If any are found, they should be penalized in your evaluation.'
            # '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Flu': '',
                   # 'Evaluation Steps:'
                   # '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
                   # '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
                   # '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
                   # '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   # 'based on the Evaluation Criteria. ',
            'Ref': ''
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its key information and primary actions of the code.'
            # '2. Read the code comments and compare them to the source code. '
            # 'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
            # '3. Check if the code comments include repetitive or unnecessary information. '
            # 'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
            # '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
        }
        rating = {
            'Coh': 'Evaluation Form (scores ONLY):',
            'Con': 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')',
            'Flu': 'Evaluation Form (scores ONLY):',
            'Ref': 'Evaluation Form (scores ONLY):',
        }
        example = {
            'Coh': [f"""
            Source Code: {coh_example['Code'].iloc[i]}
            Summary: {coh_example['Generated'].iloc[i]}
            {rating['Coh']}
            Rating: {coh_example['Coherence'].iloc[i]}""" for i in range(nshot*4)],
            'Con': [f"""
            Source Code: {con_example['Code'].iloc[i]}
            Summary: {con_example['Generated'].iloc[i]}
            {rating['Con']}
            Rating: {con_example['Consistency'].iloc[i]}""" for i in range(nshot*4)],
            'Flu': [f"""
            Source Code: {flu_example['Code'].iloc[i]}
            Summary: {flu_example['Generated'].iloc[i]}
            {rating['Flu']}
            Rating: {flu_example['Fluency'].iloc[i]}""" for i in range(nshot*4)],
            'Ref': [f"""
            Source Code: {ref_example['Code'].iloc[i]}
            Summary: {ref_example['Generated'].iloc[i]}
            {rating['Ref']}
            Rating: {ref_example['Relevance'].iloc[i]}""" for i in range(nshot*4)],
        }

    else:
        roles = {
            # coherence
            "Original Code Author 0": "As the Original Code Author, having written the code, you ensure the coherence of the code summary, ensuring that it clearly conveys the main logic "
                                      "of the code and is easy to follow.",
            # Consistency
            "Original Code Author 1": "As the Original Code Author, having written the code, you guarantee that the summary remains consistent with the original code, without hallucinated or unsupported content, similar to fact-checking to prevent any fabricated functionality.",

            # Fluency
            "Original Code Author 2": "As the Original Code Author, having written the code, you focus on ensuring that the summary is written smoothly, with clear sentences and appropriate "
                                      "wording, ensuring it reads naturally, like it was written by a fluent native speaker.",

            # Relevance
            "Code Reviewer": "As a Code Reviewer, serving as an experienced developer, you identify and preserve the most important parts of the code, avoiding unnecessary or off-topic content鈥攍ike aiming at the core message without distraction.",
        }
        evaluation_step = {
            'Coh': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its main functionality and key operations.'
            # '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
            # 'the main functionality and key operations of the code, and if they present them in a clear and '
            # 'logical order. '
            # '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Con': '',
            # 'Evaluation Steps:'
            # '1. Read the Source Code carefully and understand its main functionality and any key operations.'
            # '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
            # 'Ensure that the summary contains only statements that are present or implied in the source code.'
            # 'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
            # 'source code. If any are found, they should be penalized in your evaluation.'
            # '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Flu': '',
            'Ref': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its key information and primary actions of the code.'
            # '2. Read the code comments and compare them to the source code. '
            # 'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
            # '3. Check if the code comments include repetitive or unnecessary information. '
            # 'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
            # '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
        }

        rating = {
            'Coh': 'Evaluation Form (Do NOT give a score. Answer with exactly three lines. Line 1 starts with ``Issue:`` and states the single most important problem of the summary regarding coherence. Line 2 starts with ``Evidence:`` and gives the most relevant code evidence supporting the issue. Line 3 starts with ``Revision_goal:`` and states how the summary should be revised to improve coherence without adding unsupported facts.):',
            'Con': 'Evaluation Form (Do NOT give a score. Answer with exactly three lines. Line 1 starts with ``Issue:`` and states the single most important problem of the summary regarding consistency. Line 2 starts with ``Evidence:`` and gives the most relevant code evidence supporting the issue. Line 3 starts with ``Revision_goal:`` and states how the summary should be revised to improve consistency without adding unsupported facts.):',
            'Flu': 'Evaluation Form (Do NOT give a score. Answer with exactly three lines. Line 1 starts with ``Issue:`` and states the single most important problem of the summary regarding fluency. Line 2 starts with ``Evidence:`` and gives the most relevant code evidence or wording problem supporting the issue. Line 3 starts with ``Revision_goal:`` and states how the summary should be revised to improve fluency while preserving the original facts.):',
            'Ref': 'Evaluation Form (Do NOT give a score. Answer with exactly three lines. Line 1 starts with ``Issue:`` and states the single most important problem of the summary regarding relevance. Line 2 starts with ``Evidence:`` and gives the most relevant code evidence showing what essential information is missing or overemphasized. Line 3 starts with ``Revision_goal:`` and states how the summary should be revised to improve relevance without adding unsupported facts.):',
        }
        example = {
            'Coh': [f"""
            Source Code: {coh_example['Code'].iloc[i]}
            Summary: {coh_example['Generated'].iloc[i]}
            {rating['Coh']}
            Rating: {coh_example['Coherence'].iloc[i]}""" for i in range(nshot*5)],
            'Con': [f"""
            Source Code: {con_example['Code'].iloc[i]}
            Summary: {con_example['Generated'].iloc[i]}
            {rating['Con']}
            Rating: {con_example['Consistency'].iloc[i]}""" for i in range(nshot*5)],
            'Flu': [f"""
            Source Code: {flu_example['Code'].iloc[i]}
            Summary: {flu_example['Generated'].iloc[i]}
            {rating['Flu']}
            Rating: {flu_example['Fluency'].iloc[i]}""" for i in range(nshot*5)],
            'Ref': [f"""
            Source Code: {ref_example['Code'].iloc[i]}
            Summary: {ref_example['Generated'].iloc[i]}
            {rating['Ref']}
            Rating: {ref_example['Relevance'].iloc[i]}""" for i in range(nshot*5)],
        }
   
    df = _EVAL_DATA_OVERRIDE.copy() if _EVAL_DATA_OVERRIDE is not None else load_eval_data_from_jsonl(num)
    # Define the columns for the results DataFrame
    columns = ['sample_id', 'Code', 'Target', 'Generated'] + FOUR_DIM_COLS + CRITIC_DIM_COLS

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    # for idx, row in df.iterrows():
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], disable=_EVAL_DATA_OVERRIDE is not None):
        code_to_display = row['Code']
        target = row['Target']
        generated = row['Generated']
        # print(idx)
        # print(f"Code: {code_to_display}")
        # print(f"Reference: {target}")
        # print(f"Summary (To Be Evaluated): {generated}")
        scores_dict = {
            'sample_id': row['sample_id'],
            'Code': code_to_display,
            'Target': target,
            'Generated': generated
        }

        for (role_name, role_description), (criterion_name, criterion_task), (eval_name, eval_step), \
            (example_name, example_data),(rating_name, rating_data) in zip(roles.items(), criteria.items(), evaluation_step.items(),
                                                example.items(), rating.items()):
            if is_full_score(row[criterion_name]):
                continue
            demonstration = "\n".join(example_data)
            # demonstration = example_data
            prompt = f"""
            {role_description}
            You will be given one summary written for a source code. 
            Your task is to diagnose the most important problem of the summary on one metric and provide an actionable revision goal.
            Do NOT give a score. Only provide a structured diagnosis for revision.
            Evaluation Criteria:
            {criterion_name}(0-4) - {criterion_task}
            {eval_step}
            Example:
            {demonstration}
            Evaluate item:
            Source Code: {code_to_display}
            Summary: {generated}
            {rating_data}
            """
            score = model_api(model, prompt)
            # print(prompt)
            scores_dict[f'{criterion_name}_critic'] = score
            # Printing out the desired information:
            # print(f"Role: {role_name}")
            # print(f"Criterion: {criterion_name}")
            # print(f"Score: {score}")
        # print("------" * 10)
        # Append the result to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([scores_dict])], ignore_index=True)
    return results_df

def generate_single_critic(source_code, generated_summary, target_dimension, model, reference=0, nshot=1):
    if target_dimension not in FOUR_DIM_COLS:
        raise ValueError(f'Unknown target dimension: {target_dimension}')

    global _EVAL_DATA_OVERRIDE
    dimension_scores = {dim_name: 4.0 for dim_name in FOUR_DIM_COLS}
    dimension_scores[target_dimension] = 0.0
    _EVAL_DATA_OVERRIDE = pd.DataFrame(
        [
            {
                'sample_id': 0,
                'Code': source_code,
                'Target': '',
                'Generated': generated_summary,
                **dimension_scores,
            }
        ]
    )
    try:
        result_df = evaluate(FEWSHOT_NUM, model, reference, nshot)
    finally:
        _EVAL_DATA_OVERRIDE = None

    if result_df.empty:
        raise ValueError('No critic row returned for the candidate summary.')
    field_name = f'{target_dimension}_critic'
    value = result_df.iloc[0].get(field_name, '')
    return '' if pd.isna(value) else str(value)

def model_api(model, prompt):

    if model == 'gpt-4' or model == 'gpt-3.5-turbo':
        message = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
            )
            generated_answer = ' '.join(response.choices[0]['message']['content'].strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    elif model in CHAT_COMPLETION_MODELS:
        try:
            message = [
                      {"role": "user", "content": prompt}
               ]
            response = openai.chat.completions.create(
              model=model,
              messages=message,
              response_format={
                "type": "text"
              },
              temperature=1,
              max_completion_tokens=1000,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0,
                            store=False,
            )
            generated_answer = ' '.join(response.choices[0].message.content.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    else:
        try:
            response = openai.Completion.create(
                engine=model,  # gpt-4, gpt-3.5-turbo, text-davinci-003, text-davinci-002
                prompt=prompt,
                max_tokens=100,
            )
            generated_answer = ' '.join(response.choices[0].text.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    return generated_answer


if __name__ == '__main__':

    reference = 0  # 0-false, 1-ture

    # model = "text-davinci-003"
    # model = 'gpt-3.5-turbo'
    # model = 'gpt-4'
    model = 'gpt-4o-mini'
    turn_num = 1
    print("reference:", reference, "turns:", turn_num)
    nshot = 1


    available_rows = count_jsonl_rows(INPUT_JSONL_PATH) - START_INDEX
    if available_rows < NUM_SAMPLES:
        raise ValueError(f'Expected at least {NUM_SAMPLES} rows, but found {available_rows}.')

    # for t in range(turn_num):
    #     final_df = pd.DataFrame()
    num = NUM_SAMPLES
    df_turn = evaluate(num, model, reference, nshot)
        # last_five_columns = df_turn.iloc[:, -4:]
        # final_df = final_df.append(last_five_columns, ignore_index=True)
    write_scores_back_to_jsonl(df_turn)
    print('updated:', INPUT_JSONL_PATH)
