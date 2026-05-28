import random
import os
import pandas as pd

# 随机抽取100条，都是同样的那三百条
# Generate a list of 8714 consecutive natural numbers starting from 1
numbers = list(range(8714))

# Randomly select 300 numbers from the list
selected_numbers = random.sample(numbers, 100)

# 提取已有模型结果
approaches = ["codenn", "deepcom", "astattgru", "rencos", "ncs"]
results_df = pd.DataFrame()
for approach in approaches:
    selected_rows = []
    preds_filename = os.path.join('./dataset/TLC', approach, "random%s" % 2, "test.pred")
    f = open(preds_filename, 'r', encoding="utf-8")
    res = []
    for row in f:
        res.append(row.rstrip('\n'))
    df = pd.DataFrame(res, columns=[approach])
    selected_rows = df.loc[selected_numbers]
    results_df = pd.concat([results_df, selected_rows], axis=1)
results_df.to_excel(f"results2.xlsx", index=False)

# 提取 code 给 llms 生成摘要
df = pd.read_excel('./dataset/TLC/code.xlsx')
selected_rows = df.loc[selected_numbers]
selected_rows.to_excel(f"code2.xlsx", index=False)
