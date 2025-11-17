<<<<<<< HEAD
# ReAct Agent with Multiple Tools - Streamlit UI

This project implements a ReAct (Reasoning and Acting) agent with three tools: Search, Compare, and Analyze, with a Streamlit user interface.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Create a `.env` file in the project root directory:
   ```bash
   # On Windows (PowerShell)
   New-Item -Path .env -ItemType File
   
   # On Linux/Mac
   touch .env
   ```

2. Copy the content from `env_template.txt` or manually create the `.env` file with:
   ```
   GOOGLE_API_KEY=your_actual_google_api_key
   TAVILY_API_KEY=your_actual_tavily_api_key
   ```

3. Replace the placeholder values with your actual API keys:
   - Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Get your Tavily API key from [Tavily](https://tavily.com/)

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

- **Search Tool**: Search the web using Tavily API
- **Compare Tool**: Compare multiple items in a category
- **Analyze Tool**: Analyze and summarize search results or comparisons
- **Step-by-Step Reasoning**: View the agent's reasoning process
- **Query History**: Keep track of previous queries

## Usage

1. Enter your query in the text area
2. Click "Submit Query"
3. View the final answer and step-by-step reasoning process
4. Use example queries from the sidebar to get started

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `react_agent.py`: ReAct agent implementation with tools
- `.env`: Environment variables (create from .env.example)
- `requirements.txt`: Python dependencies

=======
# HW4_Problem3
>>>>>>> 70fa12335a7f25deb5aa3bfc8a15f01e241cbece
