import pandas as pd

# 加载数据和应用转换
df = pd.read_excel('./final/recode.xlsx')

# 创建单独的HTML文件以显示每一行的代码
for idx, row in df.iterrows():
    idx = idx + 1
    code_to_display = row['reCode']
    target = row['Target']
    generated = row['Generated']

    # 获取下一个文件的索引，上一个文件的索引
    next_idx = idx + 1
    pre_idx = idx - 1

    # 如果当前文件不是最后一个文件，创建跳转到下一个文件的按钮
    if next_idx < len(df)+1:
        next_button_html = f'<a href="code_{next_idx}.html"><button>Next</button></a>'
    else:
        next_button_html = ""  # 如果是最后一个文件，不添加按钮

    if pre_idx > 0:
        pre_button_html = f'<a href="code_{pre_idx}.html"><button>Back</button></a>'
    else:
        pre_button_html = ""  # 如果是第一一个文件，不添加按钮

    # 创建HTML内容
    html_content = f"""
        <html>
            <head>
                <title>Code Summarization Evaluation {idx}</title>
            </head>
            <body>
                <h1>Data Display {idx}</h1>
                <h2>Code:</h2>
                <pre>{code_to_display}</pre>
                <h2>Reference:</h2>
                <pre>{target}</pre>
                <h2>Generated:</h2>
                <pre>{generated}</pre>
                {pre_button_html}  <!-- 添加按钮 -->{next_button_html}  <!-- 添加按钮 -->
            </body>
        </html>
        """

    # 将HTML内容写入文件
    filename = f"./data4eval/code_{idx}.html"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(html_content)

print("HTML files created successfully.")
