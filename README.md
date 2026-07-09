# AgentProbe: LCNC Agent Automated Evaluation Framework

AgentProbe is an automated testing and multi-dimensional evaluation tool specifically designed for Low-Code/No-Code (LCNC) agents. It streamlines the entire pipeline from inputting agent profiles to generating automated test cases, executing batch simulated interactions, and exporting visual evaluation reports, significantly enhancing the production quality and optimization efficiency of your agents.

Out-of-the-box support is currently fully optimized for the Coze platform.

---

## Configuration

Create a `.env` file in the root directory of your project to configure the environment variables required for evaluation:

```env
# Your Coze Personal Access Token
COZE_PAT="your_coze_personal_access_token_here"
# The unique ID of the target agent (Bot) you wish to test
BOT_ID="your_target_agent_id_here"

# --- LLM Core Configuration ---
# The API base URL and API key of your LLM provider
API_BASE="https://api.your-provider.com/v1"
API_KEY="your_llm_api_key_here"
```

> **Note**: Make sure to add the `.env` file to your `.gitignore` to prevent any accidental leakage of your sensitive credentials.

---

## Quick Start

### 1. Installation
Ensure your local environment runs Python 3.9+. Execute the following command in the project root directory to install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Execution
Run the primary script from your terminal to launch the interactive command-line interface:

```bash
python main.py
```

### 3. Workflow & Operations

Once the system is initialized, follow the terminal prompts to perform the operations below:

#### Feature 1: Launch Automated Evaluation
1. **Input Agent Profile**: Enter the core functionality or business positioning of your current agent (e.g., `Cross-Border E-Commerce Customer Assistant`).
2. **Declare Testing Requirements**: Type in specific aspects you want to focus on for this run. If you wish to perform a comprehensive full-dimensional checkup, simply press Enter.
3. **Preview Test Plan**: The framework will automatically reverse-engineer a suite of high-quality adversarial test cases. You can preview these questions directly in the terminal.
4. **Confirm and Execute**: Type `y` or `yes` in the terminal. The system will automatically initiate high-concurrency asynchronous requests to perform batch simulation tests on your target Bot.
5. **View Evaluation Report**: After the execution completes, a professional Markdown evaluation report will be generated and saved under the `outputs/` directory.

#### Features 2-4: Rubric Evolution & Management
* **Feature 2**: If you feel the current evaluation criteria are not rigorous enough, you can input your optimization suggestions. The system will automatically interpret them and upgrade the underlying evaluation rule repository.
* **Feature 3**: Instantly roll back the evaluation rubric to the previous version with a single command.
* **Feature 4**: Reset the evaluation rubric to factory settings (revert completely to the initial default rules)
