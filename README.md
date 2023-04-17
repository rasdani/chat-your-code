# chat-your-code

Chat with your codebase.

Embeds all `.py` files in a given directory and lets you query them.
Finds cosine similiar source files and stuffs them into ChatGPT's prompt.

Export your API key as `OPENAI_API_KEY` and run `python main.py`.


TODO:
- [ ] make better use of context window
- [ ] play around with prompt
- [x] chat interactively on command line
- [ ] look into how chat history is persisted
- [x] include metadata of source files (e.g. filenames, line numbers)
