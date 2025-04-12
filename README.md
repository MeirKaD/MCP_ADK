# Web Search Agent using Google ADK and Bright Data MCP

This repository contains a web search agent built with Google's Agent Development Kit (ADK) and Bright Data's Machine Comprehension Platform (MCP). The agent can search the web and retrieve information based on user queries.

## Prerequisites

- Python 3.12 or later
- Node.js and npm (for Bright Data MCP)
- Google Gemini API key
- Bright Data account with access to MCP (Machine Comprehension Platform)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MCP_ADK.git
cd MCP_ADK
```

### 2. Create and activate a virtual environment

```bash
# For macOS/Linux
python -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install google-adk google-generativeai python-dotenv
```

### 4. Install Bright Data MCP package

```bash
npm install -g @brightdata/mcp
```

### 5. Set up environment variables

Create a `.env` file in the root directory by copying the `.env.template`:

```bash
cp .env.template .env
```

Then, edit the `.env` file and add your Google Gemini API key:

```
GOOGLE_GENAI_USE_VERTEXAI="False"
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

### 6. Configure Bright Data MCP credentials

Edit the `web_search_agent/agent.py` file and replace the placeholders with your Bright Data credentials:

```python
"API_TOKEN": "YOUR_BRIGHT_DATA_API_TOKEN",
"WEB_UNLOCKER_ZONE": "unblocker",
"BROWSER_AUTH": "brd-customer-YOUR_CUSTOMER_ID-zone-scraping_browser:YOUR_PASSWORD"
```

## Running the Agent with ADK Web Interface

### 1. Start the ADK Web Server

```bash
adk web
```

This will start a local web server, typically at `http://localhost:8000`.

### 2. Access the Web Interface

Open your browser and navigate to `http://localhost:8000` to interact with your agent through the ADK web interface.

## How the Agent Works

The agent is built using Google's Agent Development Kit (ADK) and uses Gemini 2.0 Flash as the underlying model. It leverages Bright Data's Model Context Protocol (MCP) to perform web searches and retrieve information from websites.

The agent initializes the MCP toolset asynchronously when the first request is received, connecting to Bright Data's services to enable web search capabilities.

## Features

- Web search using Bright Data MCP
- Information retrieval from websites
- Answering questions based on web content
- Automatic cleanup of resources when the agent terminates

## Customization

You can customize the agent's behavior by modifying the `web_search_agent/agent.py` file:

- Change the model by updating the `model` parameter
- Modify the agent's description and instructions
- Add additional tools or capabilities

## Troubleshooting

If you encounter issues:

1. Ensure your Google Gemini API key is valid
2. Check your Bright Data credentials
3. Verify that Node.js and npm are correctly installed
4. Make sure you have the correct version of Python and all required packages

## License

MIT

## Acknowledgements

- Google Agent Development Kit (ADK)
- Bright Data Machine Comprehension Platform (MCP)
