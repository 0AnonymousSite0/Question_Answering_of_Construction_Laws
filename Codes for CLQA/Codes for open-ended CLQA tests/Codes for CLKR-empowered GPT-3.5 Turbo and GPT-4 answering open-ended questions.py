import openai
import os
import pandas as pd
from openpyxl import load_workbook

# Set up the API key
os.environ["OPENAI_API_KEY"] = ''

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read specific column from Excel
def read_excel_column(file_path, sheet_name, column_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name].tolist()

# Read the data from Excel (adjust the file path and column names as needed)
questions_file = "100_OEQs.xlsx"  # Path to your .xlsx file
questions = read_excel_column(questions_file, "Sheet1", "问题")  # Question column
requirements = read_excel_column(questions_file, "Sheet1", "单选多选")  # Requirement column
knowledge = read_excel_column(questions_file, "Sheet1", "提示")  # Knowledge column

# Create a new Excel file and write the header (if needed)
output_file = "answers_from_openai_multi_models.xlsx"
header_written = False

if not os.path.exists(output_file):
    pd.DataFrame(columns=["Question", "Requirement", "Knowledge", "Answer1", "Answer2", "Answer3"]).to_excel(output_file, index=False)
    header_written = True

# Process each question and get the response from three models, writing after each iteration
for i in range(len(questions)):
    question_text = questions[i]
    requirement_text = requirements[i]
    knowledge_text = knowledge[i]

    # Combine the question, requirement, and knowledge into a prompt
    input_text = f"Question: {question_text}\nKnowledge: {knowledge_text}\nRequirement: {requirement_text}"

    # Get answers from three different models
    completion1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}]
    )

    completion3 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": input_text}]
    )

    # Extract answers from responses
    answer1 = completion1['choices'][0]['message']['content']

    answer3 = completion3['choices'][0]['message']['content']

    # Create a DataFrame for the current question's result
    result_df = pd.DataFrame([{
        "Question": question_text,
        "Requirement": requirement_text,
        "Knowledge": knowledge_text,
        "Answer1": answer1,

        "Answer3": answer3
    }])

    # Append the result to the Excel file without writing the header after the first time
    with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
        startrow = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
        result_df.to_excel(writer, sheet_name='Sheet1', index=False, header=not header_written, startrow=startrow)

    header_written = True  # Ensure header is not rewritten after the first row

    print(f"Processed question {i + 1}: Answer1: {answer1}, Answer3: {answer3}")

print(f"Answers saved progressively to {output_file}")
