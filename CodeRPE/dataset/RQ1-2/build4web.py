from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

df = pd.read_excel('recode.xlsx')

# 添加评分列
df['rating'] = None


@app.route('/', methods=['GET', 'POST'])
def display_data():
    idx = int(request.args.get('idx', 0))
    if idx >= len(df):
        return "All entries have been rated!"

    if request.method == 'POST':
        # 获取评分并保存
        rating = request.form.get('rating')
        df.loc[idx, 'rating'] = rating
        # 保存到Excel
        df.to_excel('align2code_rated.xlsx', index=False)
        return redirect(url_for('display_data', idx=idx + 1))

    row = df.iloc[idx]
    return render_template_string(HTML_TEMPLATE, idx=idx, code=row['reCode'], target=row['target'],
                                  generated=row['generated'])


HTML_TEMPLATE = '''
<html>
    <head>
        <title>Rate Generated Code</title>
    </head>
    <body>
        <h1>Data {{ idx }}</h1>
        <h2>Code:</h2>
        <pre>{{ code }}</pre>
        <h2>Target:</h2>
        <pre>{{ target }}</pre>
        <h2>Generated:</h2>
        <pre>{{ generated }}</pre>
        <form method="post">
            <label for="rating">Rate the generated code (1-5):</label>
            <select name="rating">
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select>
            <input type="submit" value="Submit">
        </form>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
