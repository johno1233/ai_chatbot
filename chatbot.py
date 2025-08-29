import gradio as gr
import ollama
import argparse
import subprocess
import sys
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.prompts import PromptTemplate

# parse cmd-line arguments
parser = argparse.ArgumentParser(description="Run a JARVIS-like text-based assistant with a specific model")
parser.add_argument("model", type=str, help="Name of the model (e.g., mistral, llama3.1:8b)")
args = parser.parse_args()

# Check if model is installed and pull if needed
def ensure_model_installed(model_name):
    print(f"{model_name}") # prints correct input
    try:
        # list installed models
        #print(ollama.list()["models"]) # list prints out correct list, must be how i'm searching?
        installed_models = [m["model"] for m in ollama.list()["models"]]
        print(f"{installed_models}")
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
    llm = Ollama(model=args.model)
    tools = load_tools(["ddg-search", "python_repl"], llm=llm) #DDG For web search REPL for code/tasks
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

#System prompt for JARVIS personality
JARVIS_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="You are JARVIS, Tony Stark's witty and efficient AI assistant from Iron Man. Respond helpfully, with a touch of sarcasm when appropriate. Use tools if needed for tasks like searching the web or calculations. User query: {input}"
)

# Chat function with history, agent, and error handling
def chat_with_jarvis(message, history):
    try:
        if not message:
            return "Please provide a command, sir", history
        
        # Build full prompt with history
        full_prompt = ""
        for user_msg, ai_msg in history:
             full_prompt += f"User: {user_msg}\nJARVIS: {ai_msg}\n"
        full_prompt += f"User: {message}\nJARVIS:"

        # Use agent for tool-enabled response
        agent = init_agent()
        response = agent.run(JARVIS_PROMPT.format(input=full_prompt))
        history.append((message, response))
        return response, history
    except Exception as e:
        return f"Error: Failed to generate response.\nDetails: {str(e)}.\nCheck system specs, model requirements, or if ollama is running.", history

# Ensure model is intalled
ensure_model_installed(args.model)

# launch gradio interface
with gr.Blocks(title="JARVIS Assistant") as demo:
    gr.Markdown("# JARVIS: Your AI Assistant")
    chatbot = gr.Chatbot()
    text_input = gr.Textbox(label="Type your command", placeholder="e.g., 'Search  for AI news' or 'Calculate 5 factorial'")
    submit_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    submit_btn.click(
        chat_with_jarvis,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

try:
    demo.launch()
except Exception as e:
    print(f"Error launching Gradio interface: {str(e)}")
    sys.exit(1)
