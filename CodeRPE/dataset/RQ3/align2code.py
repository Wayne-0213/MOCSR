import json
import pandas as pd
import os


def align_code_summary_ast(folder_path, df):
    # 构建文件路径
    code_file = os.path.join(folder_path, 'code.json')
    summary_file = os.path.join(folder_path, 'summary.json')
    # ast_file = os.path.join(folder_path, 'ast.json')

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
folders = ['../RQ1/dataset/TLC_Dedup']
# ['../RQ1/dataset/TLC', '../RQ1/dataset/TLC_Dedup','../RQ1/dataset/CSN','../RQ1/dataset/FCM']  # 请根据实际情况修改文件夹名
filename = './dataset/TLC_dedup/test.gold'  # 修改为您的 xlsx 文件名
f = open(filename, 'r', encoding="utf-8")
res = []
for row in f:
    res.append(row.rstrip('\n'))

# 创建 DataFrame
df = pd.DataFrame(res, columns=['Target'])
new_columns = ['Code', 'file']

for col in new_columns:
    if col not in df.columns:
        df[col] = ''

for folder in folders:
    df = align_code_summary_ast(folder, df)

# 在所有文件夹都处理完后，保存到xlsx文件
output_name = './dataset/TLC_dedup/code.xlsx'
df.to_excel(output_name, index=False)
