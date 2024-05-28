# Question_Answering_of_Construction_Laws

## !!! As the paper is under review, all contents in this repository currently are not allowed to be re-used by anyone until this announcement is deleted.

# 0. Deployed prototype of the CLQA and CLKR update
The code for deploying LLMs for CLQA in WebUI is available in this repository
![image](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/2e386b54-042d-4715-b477-92cb10e10ab8)

Please download the corresponding embedding model (bge-large-zh https://huggingface.co/BAAI/bge-large-zh-v1.5) and LLMs (ChatGLM2-6bhttps://huggingface.co/THUDM/chatglm2-6b) and put them into the folder: Codes for deploying the GPLLMs for CLQA in WebUI 
![绘图1](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/d8bb2987-73ad-4731-9e13-bd469f8a741b)
↓↓↓ CLQA deployed in the WeiUI
![A20showing20CLQA20updating20CLKR20depolyed%20prototype_converted](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/71bd62b0-0531-41f1-938f-fb228b33c143)

↓↓↓ Updating the CLKR
![A20showing20CLQA20updating20CLKR20depolyed%20prototype_converted1](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/7bfe8b6f-d64a-4d25-8207-4a0baefd0c06)

The complete demonstration video is available at: Video of a prototype showing the CLQA and CLKR update.mp4

# 1. General introduction
1.1 This repository aims to provide the codes and data regarding the paper entitled “……” developed by University of XXX in UK, The University of XXX in Hong Kong SAR, and XXX University in China for the public.

1.2 We greatly appreciate the selfless spirits of these voluntary contributors for a series of open Python libraries, including langchain (https://github.com/langchain-ai/langchain), pythonProject. https://github.com/Domengo/pythonProject/blob/master/llm-chat/langchain_gemini_qa.py, Llama (https://github.com/meta-llama/llama), ChatGLM2-6b (https://github.com/THUDM/ChatGLM2-6B), and so on.

1.3 As for anything regarding the copyright, please refer to the MIT License or contact the authors.

1.4 All of the codes have been tested to be well-performed. Even so, we are not able to guarantee their operation in other computing environments due to the differences in the Python version, computer operating system, and adopted hardware.

# 2. Summary of supplemental materials
![屏幕截图 2024-05-28 182514](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/6317a294-83d1-4de1-87d8-02da4e146dc9)


# 3. Reuse of CLKR 
The Construction Law Knowledge Repository (CLKR) follows a four-tier structure as construction law knowledge (CLK)-8 CLK areas-164 CLK subareas-387 CL documents. This repository consists of 387 documents and their relationships with 164 CLK subareas. 

More information can be found through (https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/blob/main/Table%20S2%20Construction%20law%20knowledge%20repository.xlsx)

The documents included in the CLKR are determined based on judegments from construction-related cases. If in Chinese, the use of guillemets can be selected as the identifier for documents during the mining process.
![image](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/40e9c8bf-f108-4e1c-8395-7004af727c6e)
↑↑↑Code for document name entity identification in written judgements

# 4. Reuse of CLQA validation set
The CLQA validation dataset consists of 2140 multiple-choice questions sourced from 24 test papers from the Professional Construction Engineer Qualification Examination (PCEQE) conducted between 2014 and 2023, and each question is manually labeled with paper, question type, and specific CLK area tags. 

For a more detailed version please refer to(https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/blob/main/Table%20S3%20Validation%20set%20for%20CLQA%20based%20on%2024%20test%20papers%20of%20PCQEQ.xlsx) 

# 5. Reuse of the codes for CLQA based on GPLLMs with and without CKLR
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

![image](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/d41b5de6-82a5-49c9-876f-fb62b2533b90)

↑↑↑Codes for CLQA based on original GPLLMs

![image](https://github.com/0AnonymousSite0/Question_Answering_of_Construction_Laws/assets/39326629/b9e08adc-0e17-451a-8731-1ec98f16018a)

↑↑↑Codes for CLQA based on CLKR-empowered GPLLMs


