# DS-Capstone-G14
Capstone project individual repository. 

Links for the tasks and individual explorations done before and during the initial stages of the project are listed below: 
- UI exploration on gradio: https://github.com/Monissha/gradio-integration-for-ai-chatbot
- OCR(PDF to Text): https://github.com/Monissha/Convert-PDF-files-to-OCR

1. The rest of the tasks and experiments that I underwent during the entire project, mostly consists of bash scripts and is done on command prompt 80% of the time. Trying to optimize and accelerate the fine-tuning and bulk processing files which require high GPU resources, have all been done through setonix(pawsey supercomputer). This enabled the team to accelerate to next steps without exhausting their GPU limitations and available resources. Processes like embedding and vectorisation is also done using setonix as the dataset here is a huge PDF file that needed OCR for extraction. Furthermore, fine-tuned LLM models were also run in setonix for faster processing and GPU utilization, which required cleaning up the code and installing necessary packages in the said virutal environment, to facilitate smooth running and fine-tuning processes. 

2. UI of the final model chatbot has been done through a python package called gradio. I have also made custom chat interface to enable smooth processing and better user experience. Integrating this UI interface with our RAG model required substantial understanding in the RAG prototyping and prompt engineering. The code for the UI with RAG integrated with it, is included in this repository. 

3. Dockerization of the entire project for future usage, has also been done. This required creating dockerfiles and setting up a directory and pulling GPU images like nvidia cuda explicitly. Additionally, it is required that I change the notebook file into a python file with only the necessary code, cleaning it up to smoothly interact with the docker. Requirements file has also been included. Dockerfile primarily involves downloading various necessary packages, and setting up an environment for the fine-tuned model integrated with RAG and UI, to run. Although it definitely required enormous GPU and CPU resources as it is running on the local device, the docker image still works if run on a high performing system with higher RAM and good GPU. 
