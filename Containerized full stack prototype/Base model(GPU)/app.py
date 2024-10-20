from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
from IPython.display import display, Markdown  # For displaying markdown in the interface


# Use precomputed embeddings stored locally
vectorstore = Chroma(
    persist_directory="/app/vector1",  # Directory inside the Docker container
    embedding_function=None  # No need to recompute embeddings
)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant documents


# Initialize the language model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000)
llm = HuggingFacePipeline(pipeline=pipe)



# Define the RAG Chat Model class
class RAGChatModel:
    def __init__(self, retriever, llm, tokenizer, max_token_limit=1200):
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_token_limit = max_token_limit
        self.current_token_count = 0
        self.template_standard = """
        <|system|>
        Answer the question and all the page numbers where this information is found based in the information provided in the context.
        Providing all the relevant page numbers is essential.

        Context: {context}

        Providing all the relevant page numbers is essential.
        <|end|>

        <|user|>
        Question: {question}
        <|end|>

        <|assistant|>
        """
        self.template_exceeded = """
        <|system|>
        Answer the question in detail; warn that information is not taken from the prescribed textbook and provide the page numbers where they can find the correct information in the prescribed textbook.

        Context: {context}
        Providing all the relevant page numbers is essential.
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
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Generate prompt based on token count
        prompt = self.get_prompt(docs, question)

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
    # For simplicity, return the first few words of the question
    short_question = (question[:max_length] + '...') if len(question) > max_length else question
    return short_question

# Function for the RAG model interaction
def ask_question_gradio(history, question):
    """Main function to retrieve relevant docs and generate a response."""
    # Retrieve relevant documents
    docs = rag_chat_model.retriever.invoke(question)

    # Generate prompt based on token count
    prompt = rag_chat_model.get_prompt(docs, question)

    # Pass the prompt to the LLM
    result = rag_chat_model.llm.generate([prompt])

    # Extract the generated text
    raw_answer = result.generations[0][0].text

    # Get the clean answer
    clean_answer = rag_chat_model.extract_clean_answer(raw_answer)

    # Add the question and answer to the conversation history
    history.append((question, clean_answer))

    # Generate a short summary for the chat history section (only from the question)
    short_overview = get_short_overview(question, clean_answer)

    # Append the short overview to the chat history display (only questions' summaries)
    chat_history = "\n\n".join([get_short_overview(q, a) for q, a in history])

    # Return the updated history, overview history for the left panel, and clear the user input
    return history, chat_history, ""  # The empty string clears the input box

# Create a Gradio Blocks interface for chat-like interaction
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>RAG Model Chat</h1>
        <p style='text-align: center;'>Ask any question and get a response from the RAG model.</p>
        """,
    )

    with gr.Row():
        # Sidebar for chat history
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Chat History Overview")
            history_display = gr.Textbox(label="Chat History", lines=20, interactive=False)  # Non-editable

        # Main Chatbot Area
        with gr.Column(scale=2):
            # Chatbot layout with conversation history
            chatbot = gr.Chatbot(label="Chat with RAG Model")

            # Text input for user questions
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask your question...",
                    label="Type your message here:"
                )

                # Button to submit the question
                submit_button = gr.Button("Send")

            # Maintain the conversation history
            history_state = gr.State([])

            # Link the components: send input, update chatbot, clear textbox, and update history display
            submit_button.click(
                ask_question_gradio,
                inputs=[history_state, user_input],
                outputs=[chatbot, history_display, user_input],
                scroll_to_output=True
            )
            user_input.submit(
                ask_question_gradio,
                inputs=[history_state, user_input],
                outputs=[chatbot, history_display, user_input],
                scroll_to_output=True
            )

# Launch the Gradio interface
demo.launch(share=True)

