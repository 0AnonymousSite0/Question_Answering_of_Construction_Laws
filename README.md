# Question_Answering_of_Construction_Laws

## !!! As the paper is under review, all contents in this repository currently are not allowed to be re-used by anyone until this announcement is deleted.

# 0. Summary of supplemental materials
![屏幕截图 2024-05-28 182514](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/6317a294-83d1-4de1-87d8-02da4e146dc9)
# 1. general introduction
1.1 This repository aims to provide the codes and data regarding the paper entitled “……” developed by University of XXX in UK, The University of XXX in Hong Kong SAR, and XXX University in China for the public.
1.2 We greatly appreciate the selfless spirits of these voluntary contributors for a series of open Python libraries, including langchain (https://github.com/langchain-ai/langchain), pythonProject. https://github.com/Domengo/pythonProject/blob/master/llm-chat/langchain_gemini_qa.py, Llama (https://github.com/meta-llama/llama), ChatGLM2-6b (https://github.com/THUDM/ChatGLM2-6B), and so on.
1.3 As for anything regarding the copyright, please refer to the MIT License or contact the authors.
1.4 All of the codes have been tested to be well-performed. Even so, we are not able to guarantee their operation in other computing environments due to the differences in the Python version, computer operating system, and adopted hardware.
# 2 A prototype for the CLQA and

The code for deploying LLMs for CLQA in WebUI is available in this repository


Please download the corresponding embedding model (bge-large-zh https://huggingface.co/BAAI/bge-large-zh-v1.5) and LLMs (ChatGLM2-6bhttps://huggingface.co/THUDM/chatglm2-6b) and put them into the folder
 

the CLKR update


# 3 Reuse of CLKR 
The Construction Law Knowledge Repository (CLKR) follows a four-tier structure as construction law knowledge (CLK)-8 CLK areas-164 CLK subareas-387 CL documents. This repository consists of 387 documents and their relationships with 164 CLK subareas. 


More information can be found through this link 

The documents included in the CLKR are determined based on judegments from construction-related cases. If in Chinese, the use of guillemets can be selected as the identifier for documents during the mining process.

↑↑↑Code for document name entity identification in written judgements

# 4 Reuse of CLQA validation set
The CLQA validation dataset consists of 2140 multiple-choice questions sourced from 24 test papers from the Professional Construction Engineer Qualification Examination (PCEQE) conducted between 2014 and 2023, and each question is manually labeled with paper, question type, and specific CLK area tags. 


For a more detailed version please refer to() 

# 5 Reuse of the codes for CLQA based on GPLLMs with and without CKLR
5.1 Environment set
All codes are developed on Python 3.9, and the IDE adopted is PyCharm
aiohttp==3.9.0
aiolimiter==1.1.0
beautifulsoup4==4.12.2
bs4==0.0.1
click==8.1.7
colorama==0.4.6
filelock==3.13.1
filetype==1.2.0
flatbuffers==23.5.26
frozenlist==1.4.0
fsspec==2023.10.0
future==0.18.3
grpcio==1.59.3
h11==0.14.0
httpcore==1.0.2
httptools==0.6.1
httpx==0.25.1
Please refer to the supplementary materials for the complete requirement file.

5.2 Codes for testing the GLMs
Closed-source GPLLMs (e.g., text-davinci-003, GPT-3.5 turbo, and GPT-4) are API-only, while open-source GLMs over 24GB also use APIs to avoid high-end GPU costs.

↑↑↑Codes for CLQA based on original GPLLMs

↑↑↑Codes for CLQA based on CLKR-empowered GPLLMs


