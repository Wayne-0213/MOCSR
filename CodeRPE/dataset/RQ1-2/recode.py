import pandas as pd

def java_trans(code_str):

    code_str = str(code_str).split(' ')
    res = ""
    count = 0
    for item in code_str:
        res += item
        if (item == "{"):
            count = count + 1
            res += "\n" + "\t" * count
        elif (item == "}"):
            count = count - 1
            res += "\n" + "\t" * count
        elif (item == ";"):
            res += "\n" + "\t" * count
        else:
            res += " "
    print(res)
    return res

df = pd.read_excel('align2code.xlsx')
df['reCode'] = df['Code'].apply(java_trans)
df.to_excel('recode.xlsx', index=False)
