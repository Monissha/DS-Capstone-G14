from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from peft import PeftModel, PeftConfig
import gradio as gr
from IPython.display import display, Markdown
import torch

# Function to calculate the total number of tokens in the vector database
def count_total_tokens_in_vectorstore(vectorstore, tokenizer):
    all_docs = vectorstore.get()['documents']
    total_tokens = 0
    for doc in all_docs:
        tokens_in_doc = len(tokenizer.encode(doc))
        total_tokens += tokens_in_doc
    return total_tokens

# Initialize embeddings
embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_model_kwargs = {"device": "cpu"}
embedding_encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

# Initialize vector store and retriever
vectorstore = Chroma(
    persist_directory="/app/vector1",  # Adjust this path as needed
    embedding_function=None
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

device = "cpu"

print(f"Using device: {device}")


# Load the base model and tokenizer
base_model = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load the fine-tuned Phi3 mini model with LoRA
model = AutoModelForCausalLM.from_pretrained(base_model, return_dict=True, device_map=device, torch_dtype=torch.float32)  # Changed device_map to CPU and dtype to float32
lora_model = PeftModel.from_pretrained(model, "ShilpaSandhya/phi3_5_mini_lora_chemical_eng")

pipeline = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, max_new_tokens=count_total_tokens_in_vectorstore(vectorstore, tokenizer)//10)  # Added device=0 to enforce CPU usage
llm = HuggingFacePipeline(pipeline=pipeline)

# Define the RAG Chat Model class
class RAGChatModel:
    def __init__(self, retriever, llm, tokenizer, max_token_limit=count_total_tokens_in_vectorstore(vectorstore, tokenizer)//10):
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_token_limit = max_token_limit
        self.current_token_count = 0
        self.template_standard = """
        <|system|>
        Answer the question in detail. Provide all the relevant information based on the provided context.
        It is critical that you mention all page numbers where this information is found. Do not skip any page numbers.


        Context: {context}

        Providing all the page numbers is essential  for the answer.
        <|end|>

        <|user|>
        Question: {question}
        <|end|>

        <|assistant|>
        """
        self.template_exceeded = """
        <|system|>
        Answer the question in detail; warn that information is not taken from the prescribed textbook and must provide the page numbers where they can find the correct information in the prescribed textbook.

        Context: {context}
        Providing all the page numbers is essential for the answer.
        <|end|>

        <|user|>
        Question: {question}
        <|end|>

        <|assistant|>
        """

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string using the tokenizer."""
        return len(self.tokenizer.encode(string))

    def format_docs(self, docs, full_content=True):
        """Format the documents to be used as context in the prompt."""
        if full_content:
            return "\n\n".join(f"Information in Page number: {(doc.metadata['page']+1)}\n{doc.page_content}" for doc in docs)
        else:
            return "Information available in prescribed textbook " + ", ".join(f"Page number: {doc.metadata['page']}" for doc in docs)

    def get_prompt(self, docs, question):
        """Generate the prompt based on token count and context formatting."""
        # Format the context with full content
        context = self.format_docs(docs, full_content=True)
        total_tokens_in_context = self.num_tokens_from_string(context)

        # Add tokens to the running total
        self.current_token_count += total_tokens_in_context

        # Decide whether to use full content or only page numbers
        if self.current_token_count > self.max_token_limit:
            print("Token limit exceeded. Information from prescribed textbook will not be used.")
            # Reformat context to include only page numbers
            context = self.format_docs(docs, full_content=False)
            template = self.template_exceeded
        else:
            template = self.template_standard

        # Create the prompt
        prompt = template.format(context=context, question=question)
        return prompt

    def extract_clean_answer(self, raw_output):
        """Extract only the answer from the raw output."""
        assistant_tag = "<|assistant|>"
        if assistant_tag in raw_output:
            clean_answer = raw_output.split(assistant_tag)[-1].strip()
            return clean_answer
        return raw_output.strip()

    def ask_question(self, question):
        """Main function to retrieve relevant docs and generate a response."""
        # Add fixed request for page numbers to the user's question
        question_with_page_request = f"{question}. Please provide the page numbers in your answer."

        # Retrieve relevant documents
        docs = self.retriever.invoke(question_with_page_request)

        # Generate prompt based on token count
        prompt = self.get_prompt(docs, question_with_page_request)

        # Pass the prompt to the LLM
        result = self.llm.generate([prompt])

        # Extract the generated text
        raw_answer = result.generations[0][0].text

        # Get the clean answer
        clean_answer = self.extract_clean_answer(raw_answer)

        # Display the answer
        display(Markdown(clean_answer))


# Initialize the RAGChatModel
rag_chat_model = RAGChatModel(retriever, llm, tokenizer)





# Function to shorten the question for the chat history display
def get_short_overview(question, answer, max_length=50):
    """Generate a short summary of the question for the chat history."""
    return (question[:max_length] + '...') if len(question) > max_length else question

# Function for the RAG model interaction
def ask_question_gradio(history, question):
    """Main function to retrieve relevant docs and generate a response."""
    if not question:  # Check if the question is empty
        return history, "", ""  # Return empty if no question is asked

    # Add fixed request for page numbers to the user's question
    question_with_page_request = f"{question}. Please provide the page numbers in your answer."

    # Retrieve relevant documents using the RAG model
    docs = rag_chat_model.retriever.invoke(question_with_page_request)
    prompt = rag_chat_model.get_prompt(docs, question_with_page_request)

    # Generate the response
    result = rag_chat_model.llm.generate([prompt])
    raw_answer = result.generations[0][0].text
    clean_answer = rag_chat_model.extract_clean_answer(raw_answer)

    # Add the question and answer to the conversation history as dicts
    history.append({"role": "user", "content": question})  # Add user question
    history.append({"role": "assistant", "content": clean_answer})  # Add model answer

    # Generate a short summary for the chat history section
    short_overview = get_short_overview(question, clean_answer)
    
    # Format the chat history for display in the overview
    chat_history = "\n\n".join([f"{get_short_overview(q['content'], a['content'])}" for q, a in zip(history[::2], history[1::2])])

    # Return the updated history and chat history for display
    return history, chat_history, ""  # The empty string clears the input box

# Create Gradio Blocks interface
with gr.Blocks() as demo:
    
    gr.Markdown(
        """
        <h1 style='text-align: center;'>L-ChemNerd</h1>
        <p style='text-align: center;'>Ask any question and get a response from the RAG model.</p>
        """,
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Chat History Overview")
            history_display = gr.Textbox(label="Chat History", lines=20, interactive=False)  # Non-editable

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Lora ChemNerd Chat", type='messages')  # Ensure type is 'messages'
            user_input = gr.Textbox(placeholder="Ask your question...", label="Type your message here:")
            submit_button = gr.Button("Send")

            history_state = gr.State([])

            submit_button.click(
                ask_question_gradio,
                inputs=[history_state, user_input],
                outputs=[chatbot, history_display, user_input],  # Update chatbot and chat history
                scroll_to_output=True
            )

            user_input.submit(
                ask_question_gradio,
                inputs=[history_state, user_input],
                outputs=[chatbot, history_display, user_input],  # Update chatbot and chat history
                scroll_to_output=True
            )


# Launch the Gradio interface
demo.launch(share=True)


