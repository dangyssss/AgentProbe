# AgentProbe: LCNC Agent Automated Evaluation Framework

AgentProbe is an automated testing and evaluation framework designed for Low-Code/No-Code (LCNC) agents. The framework addresses the evaluation gap faced by end-user developers by streamlining the workflow from requirement input to quantitative performance reporting. 

While the core architecture of AgentProbe is platform-agnostic and fully extensible for migration to other LCNC platforms, the current implementation provides out-of-the-box support for the Coze platform.

## Features

* **Automated Test Planning**: Generates multi-dimensional test plans derived directly from user-provided agent descriptions and testing requirements.
* **Interactive Verification**: Displays the pre-test planning report in the terminal, allowing the user to verify and approve the test suite before execution.
* **Structured Evaluation Reporting**: Executes the test cases automatically upon user confirmation and outputs a comprehensive evaluation report containing quantitative metrics and qualitative analysis.

---

## Configuration

Before executing AgentProbe, you must configure the environment variables for both platform authentication and the underlying multi-agent execution framework.

1. Create a `.env` file in the root directory of the project.
2. Define the required credentials within the file:

```env
# --- Coze Platform Configuration ---
# Your Personal Access Token (PAT) from Coze
COZE_PAT="your_coze_personal_access_token_here"
# The specific ID of the target bot/agent you wish to audit
BOT_ID="your_target_agent_id_here"

# --- MetaGPT Core Configuration ---
# The underlying Multi-Agent framework relies on MetaGPT; configure your LLM provider here
API_BASE="https://api.your-provider.com/v1"
API_KEY="your_llm_api_key_here"
```

*Note: Ensure the `.env` file is excluded from version control to protect your sensitive credentials.*

---

## Usage

### 1. Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Execution
Run the primary script from your terminal:

```bash
python main.py
```

### 3. Workflow Sequence
Once initialized, AgentProbe guides you through the following pipeline via the command-line interface:

* **Input Phase**: The terminal prompts you to enter the functional description of your target agent along with your specific testing requirements.
* **Planning Phase**: AgentProbe analyzes the inputs to generate a multi-dimensional test plan, halting to request user confirmation.
* **Testing Phase**: Upon receiving confirmation (e.g., `Y`), the program dispatches the automated test suite across the platform APIs.
* **Reporting Phase**: After test execution completes, the final agent evaluation report is rendered directly in the terminal, with raw logs concurrently persisted to disk.