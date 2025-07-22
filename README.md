# Project
Wanted to mess around with AI. This project is a simple, locally hosted AI chat bot using python and Ollama.

# Current State
Currently, when you run the code `python chatbot.py mistral:latest` the program will check to see if you have already pulled mistral. If you have, it will start it and launch  a local gradio instance at 127.0.0.1:7860. If you don't have the agent pulled from Ollama, it will pull it and then launch the agent and a gradio interface.

# Future Plans
- I want to add hardware detection so that you can attempt to load a model
  -  If your hardware meets the recommended specs: it will pull the model and run it.
  -  If your hardware doesn't meet the specs: it will throw an error and advise you not to install it with the option to override if desired
# To run
- pull repo
- Start python env: `\chatbot\Scripts\Activate.ps1`
- In python env, run: `python chatbot.py [model name (i.e., mistral:latest)]`

