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
    Knowledge = read_excel_column(f"{year}.xlsx", "Sheet1", "提示")
    Requirements = read_excel_column(f"{year}.xlsx", "Sheet1", "单选多选")  # New column for single/multiple choice

    # Create or open the Excel file to write progressively
    with pd.ExcelWriter(f"answer_from_4_models_with_knowledge_{year}.xlsx", mode='w', engine='openpyxl') as writer:
        for i in range(len(Questions)):
            question_text = Questions[i]
            knowledge_text = Knowledge[i]
            requirement_text = Requirements[i]

            # Combine the question, knowledge, and requirement into the input
            input_text = f"Question: {question_text}\nKnowledge: {knowledge_text}\nRequirement: {requirement_text}"

            # Model responses
            ERNIE_40_8k_chat_comp_resp = ERNIE_40_8k_chat_comp.do(model="ERNIE-4.0-8K", messages=[{
                "role": "user",
                "content": input_text
            }])

            Qianfan_Chinese_Llama_2_70B_resp = Qianfan_Chinese_Llama_2_70B_chat_comp.do(
                model="Qianfan-Chinese-Llama-2-70B", messages=[{
                    "role": "user",
                    "content": input_text
                }], temperature=0.01, top_p=0)

            ChatGLM2_6B_32K_resp = ChatGLM2_6B_32K_chat_comp.do(
                model="ChatGLM2-6B-32K", messages=[{
                    "role": "user",
                    "content": input_text
                }])
            ERNIE_35_8K_resp = ERNIE_35_8K_chat_comp.do(
                model="ERNIE-3.5-8K", messages=[{
                    "role": "user",
                    "content": input_text
                }])

            # Create a DataFrame for the current question's results
            df2 = pd.DataFrame([{
                "Question": question_text,
                "Knowledge": knowledge_text,
                "Requirement": requirement_text,
                "Answer1": ERNIE_40_8k_chat_comp_resp['result'],

                "Answer3": Qianfan_Chinese_Llama_2_70B_resp['result'],

                "Answer5": ChatGLM2_6B_32K_resp['result'],
                "Answer6": ERNIE_35_8K_resp['result']
            }])

            # Write the result to the Excel sheet incrementally
            if i == 0:
                # Write the header only once
                df2.to_excel(writer, sheet_name='Sheet1', index=False, header=True)
            else:
                # Append without the header for subsequent writes
                df2.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

            print(f"Processed question {i + 1} with answers from all models.")
