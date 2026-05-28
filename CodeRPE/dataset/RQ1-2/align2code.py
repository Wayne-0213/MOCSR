import json
import pandas as pd
import os

def align_code_summary_ast(folder_path, df):
    # 构建文件路径
    code_file = os.path.join('./dataset/'+folder_path, 'code.json')
    summary_file = os.path.join('./dataset/'+folder_path, 'summary.json')
    # ast_file = os.path.join('./dataset/'+folder_path, 'ast.json')

    # 读取 JSON 文件
    code_data = []
    with open(code_file, 'r') as file:
        for line in file:
            code_data.append(' '.join(json.loads(line)))

    summary_data = []
    with open(summary_file, 'r') as file:
        for line in file:
            summary_data.append(' '.join(json.loads(line)))

    # ast_data = []
    # with open(ast_file, 'r') as file:
    #     for line in file:
    #         ast_data.append(json.loads(line))

    for index, row in df.iterrows():
        target = str(row['Target']).lstrip()

        if target in summary_data:
            code_index = summary_data.index(target)
            df.at[index, 'Code'] = code_data[code_index]
            # df.at[index, 'AST'] = ast_data[code_index]
            df.at[index, 'file'] = folder_path
    return df

# 列出您的文件夹名称
# folders = ['CSN','FCM','TLC', 'TLC_Dedup']  # 请根据实际情况修改文件夹名
folders = ['test']  # 请根据实际情况修改文件夹名
xlsx_file = 'human_evaluation_1.xlsx'  # 修改为您的 xlsx 文件名

# 初始化一个新的DataFrame来保存所有处理后的数据
df = pd.read_excel(xlsx_file)
new_columns = ['Code', 'file']

for col in new_columns:
    if col not in df.columns:
        df[col] = ''

for folder in folders:
    df = align_code_summary_ast(folder, df)

# 在所有文件夹都处理完后，保存到xlsx文件
output_name = 'align2code_2.xlsx'
df.to_excel(output_name, index=False)

