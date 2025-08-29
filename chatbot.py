import gradio as gr
import ollama
import subprocess
import sys
import json
import os
from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits.load_tools import load_tools 
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

MODEL = "mistral"

# Check if model is installed and pull if needed
def ensure_model_installed(model_name):
    print(f"Checking for model: {model_name}")
    try:
        installed_models = [m["model"] for m in ollama.list()["models"]]
        # print(f"{installed_models}")
        if model_name not in installed_models:
            print(f"Model {model_name} not found. Pulling from Ollama...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Model {model_name} successfully pulled.")
        else:
            print(f"Model {model_name} already installed")
    except subprocess.CalledProcessError:
        print(f"Error: Failed to pull {model_name}. Check connection or Ollama installation.")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking or pulling model: {str(e)}")
        sys.exit(1)

def init_agent():
    try:
        print(f"Initializing OllamaLLM with model: {MODEL}")
        llm = OllamaLLM(model=MODEL)
        print("Loading tools: ddg-search, llm-math")
        tools = load_tools(["ddg-search", "llm-math"], llm=llm)
        print("Initializing agent...")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            memory=memory
        )
        print("Agent initialized successfully.")
        return agent
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        raise

#System prompt for JARVIS personality
JARVIS_PROMPT = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="You are JARVIS, Tony Stark's witty and efficient AI assistant from Iron Man. Respond helpfully, with a touch of sarcasm when appropriate. Use tools if needed for tasks like searching the web or calculations. \n\nChat History:\n{chat_history}\n\nUser query: {input}"
)
# Fall back prompt for direct ollama chat
FALLBACK_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="You are JARVIS, Tony Stark's witty and efficient AI assistant from Iron Man. Respond helpfully, with a touch of sarcasm when appropriate. If the query requires a web search or calculation, provide a best-effort response based on your knowledge. User query: {input}")

# Feedback file
FEEDBACK_FILE = "feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

# Save feedback
def save_feedback(feedback_list):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=4)

# Chat function with history, agent, and error handling
def chat_with_jarvis(message, history):
    try:
        if not message:
            return history + [{"role": "assistant", "content": "Please provide a command, sir."}]

        print(f"Processing message: {message}")
        # Build chat_history string from Gradio history
        chat_history = ""
        for msg in history:
            if msg["role"] == "user":
                chat_history += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                chat_history += f"JARVIS: {msg['content']}\n"

        # Heuristic to check if tools might be needed (customize as needed)
        needs_tools = any(word in message.lower() for word in ["search", "calculate", "math", "find", "what is", "how many"])
        
        response = ""
        if needs_tools:
            # try agent for tool-enabled response
            try:
                agent = init_agent()
                # Format prompt with history and current input
                formatted_input = JARVIS_PROMPT.format(chat_histor=chat_history, input=message)
                response = agent.invoke({"input": formatted_input})["output"]
            except Exception as agent_error:
                print(f"Agent Failed: {str(agent_error)}")

        if not response:
            print("Using direct Ollama chat.")
            messages = [{"role": "system", "content": "Your are JARVIS, Tony Stark's witty and efficient AI assistant from Iron Man. Respond helpfully, with a touch of sarcasm when appropriate."}]
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": message})
            response = ollama.chat(model=MODEL, messages=messages)["message"]["content"]

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        print(f"Response generated: {response}")
        return history
    except Exception as e:
        error_msg = f"ERROR: Failed to generate response. Details: {str(e)}. Check system specs, model requirements, ofr if Ollama is running"
        print(error_msg)
        return history + [{"role": "assistant", "content": error_msg}]
        

# Feedback handlers
def thumbs_up(history):
    if len(history) >= 2:
        interaction = {
            "user": history[-2]["content"],
            "assistant": history[-1]["content"],
            "feedback": "positive"
        }
        feedback = load_feedback()
        feedback.append(interaction)
        save_feedback(feedback)
        return "Thanks for the feedback, sir, I'm clearly on the right track."
    return "No recent interaction to feedback on"

def thumbs_down(history):
    if len(history) >=2:
        interaction = {
            "user": history[-2]["content"],
            "assistant": history[-1]["content"],
            "feedback": "negative"
        }
        feedback = load_feedback()
        feedback.append(interaction)
        save_feedback(feedback)
        return "Apologies, sir. I'll endeavor to do better next time."
    return "No recent interaction to feedback on."

# Ensure model is intalled
ensure_model_installed(MODEL)

# launch gradio interface
with gr.Blocks(title="JARVIS Assistant") as demo:
    gr.Markdown("# JARVIS: Your AI Assistant")
    chatbot = gr.Chatbot(type="messages")
    text_input = gr.Textbox(label="Type your command", placeholder="e.g., 'Search  for AI news' or 'Calculate 5 factorial'")
    
    with gr.Row():
        submit_btn = gr.Button("Send")
        thumbs_up_btn = gr.Button("👍")
        thumbs_down_btn = gr.Button("👎")
    feedback_output = gr.Textbox(label="Feedback Status", interactive=False)
    clear_btn = gr.Button("Clear Chat")
    
    submit_btn.click(
        chat_with_jarvis,
        inputs=[text_input, chatbot],
        outputs=[chatbot]
    )
    thumbs_up_btn.click(
        thumbs_up,
        inputs=[chatbot],
        outputs=[feedback_output]
    )
    thumbs_down_btn.click(
        thumbs_down,
        inputs=[chatbot],
        outputs=[feedback_output]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

try:
    print("Launching Gradio interface...")
    demo.launch()
except Exception as e:
    print(f"Error launching Gradio interface: {str(e)}")
    sys.exit(1)
