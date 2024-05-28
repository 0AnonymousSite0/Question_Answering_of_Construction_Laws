# collecting CrLDs' name from CrCJs
import pandas as pd
import re

# importing an excel file
input_file_path = r"input.xlsx"
output_file_path = r"output.xlsx"
df = pd.read_excel(input_file_path)
column_name = "说理部分"
input_text = df[column_name]
input_text = input_text.replace("\n", "")

# Defining regular expression pattern.
pattern = r"(《[^》]*》)"
pattern1 = r"第[零一二三四五六七八九十百千万亿]+条"

# Processing the text and storing the results in a new column.
df["Case Basis 1"] = input_text.apply(lambda x: dict(zip(re.split(pattern, x)[1::2], re.split(pattern, x)[2::2])))

def match_laws(value):
    matched_laws = {}
    for key, item in value.items():
        matches = re.findall(pattern1, item)
        if matches:
            matched_laws[key] = matches
    return matched_laws

# Performing secondary matching on the values in the column of Case Basis 1.
df["Case Basis 2"] = df["Case Basis 1"].apply(match_laws)
df["Case Basis 2"] = df["Case Basis 2"].apply(lambda x: {k: v for k, v in x.items() if v})
df["Case Basis 3"] = df["Case Basis 2"].apply(lambda x: ', '.join([f'{key}{value}' for key, values in x.items() for value in values]))

df.to_excel(output_file_path, index=False)





