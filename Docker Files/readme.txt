Docker image names: 
1. Base model + RAG + UI = monissha/ragmodel
2. Finetuned LLM + RAG + UI = monissha/rag_finetuned

Docker pull and run commands: 
pull: 
docker pull monissha/image_name:latest
run:
docker run -p 7860:7860 monissha/image_name:latest
