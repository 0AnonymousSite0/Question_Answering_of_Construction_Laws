import os
import qianfan
import pandas as pd

os.environ['QIANFAN_AK'] = ""
os.environ['QIANFAN_SK'] = ""

def read_excel_column(file_path, sheet_name, column_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name].tolist()

# Instantiate models
ERNIE_40_8k_chat_comp = qianfan.ChatCompletion()
ERNIE_40_Turbo_8K_chat_comp = qianfan.ChatCompletion()
Qianfan_Chinese_Llama_2_70B_chat_comp = qianfan.ChatCompletion()
Llama_2_70b_chat_comp = qianfan.ChatCompletion()
ChatGLM2_6B_32K_chat_comp = qianfan.ChatCompletion()
ERNIE_35_8K_chat_comp = qianfan.ChatCompletion()

years = ["100_OEQs"]

for year in years:
    print("Processing examination data for", year)
    Questions = read_excel_column(f"{year}.xlsx", "Sheet1", "问题")

    # Create or open the Excel file in append mode
    with pd.ExcelWriter(f"Case_answer_from_4_models_{year}.xlsx", mode='w', engine='openpyxl') as writer:
        for i in range(len(Questions)):
            question_text = Questions[i]

            # Get responses from different models
            ERNIE_40_8k_chat_comp_resp = ERNIE_40_8k_chat_comp.do(model="ERNIE-4.0-8K", messages=[{
                "role": "user",
                "content": question_text
            }])

            Qianfan_Chinese_Llama_2_70B_resp = Qianfan_Chinese_Llama_2_70B_chat_comp.do(
                model="Qianfan-Chinese-Llama-2-70B", messages=[{
                    "role": "user",
                    "content": question_text
                }], temperature=0.01, top_p=0)

            ChatGLM2_6B_32K_resp = ChatGLM2_6B_32K_chat_comp.do(
                model="ChatGLM2-6B-32K", messages=[{
                    "role": "user",
                    "content": question_text
                }])
            ERNIE_35_8K_resp = ERNIE_35_8K_chat_comp.do(
                model="ERNIE-3.5-8K", messages=[{
                    "role": "user",
                    "content": question_text
                }])

            # Create a DataFrame for the current question's results
            df2 = pd.DataFrame([{
                "Question": question_text,
                "Answer1": ERNIE_40_8k_chat_comp_resp['result'],

                "Answer3": Qianfan_Chinese_Llama_2_70B_resp['result'],

                "Answer5": ChatGLM2_6B_32K_resp['result'],
                "Answer6": ERNIE_35_8K_resp['result']
            }])

            # Write the result to the Excel file incrementally
            df2.to_excel(writer, sheet_name='Sheet1', index=False, header=not writer.sheets)

            print("Processed question:", i + 1)
