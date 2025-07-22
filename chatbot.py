import gradio as gr
import ollama
import argparse
import subprocess
import sys
#import psutil
#import torch

# parse cmd-line arguments
parser = argparse.ArgumentParser(description="Run a chatbot with a specific model")
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

# Chat function, now with error handling
def chat_with_mistral(message, hsitory):
    try:
        response=ollama.chat(
                model=args.model,
                messages=[{"role": "user", "content": message}]
        )
        return response ["message"]["content"]
    except Exception as e:
        return f"Errro: Failed to run model {args.model}. Check system specs and model specs requirements. Check if model was installed correctly. Details: {str(e)}"

# Ensure model is intalled
ensure_model_installed(args.model)

# launch gradio interface
try:
    gr.ChatInterface(chat_with_mistral).launch()
except Exceptio as e:
    print(f"Error launching Gradio interface: {str(e)}")
    sys.exit(1)

