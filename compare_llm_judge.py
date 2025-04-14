import streamlit as st
import requests
import json
import pandas as pd
import time

# --- Configuration ---
# Fine-tuned Ollama model details
FINE_TUNED_MODEL_UI_NAME = "Ollama Crossfire (Local)"
OLLAMA_MODEL_TAG = "crossfire_model_v4_mac_v2:latest" # Make sure this tag exists locally
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama URL

# Evaluation LLM configuration (using OpenRouter)
EVALUATION_MODEL_NAME = "meta-llama/llama-4-maverick" # Or another capable model like Claude 3.5 Sonnet, GPT-4o etc.
# Check OpenRouter model availability if changing this

# List of OpenRouter models for comparison
OPENROUTER_MODELS = [
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-8b",
    "google/gemini-flash-1.5",
    "google/gemini-2.5-pro-preview-03-25",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4-turbo",
    "meta-llama/llama-4-scout",
    "deepseek/deepseek-v3-base:free",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    ]
# Ensure Evaluation model is not duplicated if it's already in the list
if EVALUATION_MODEL_NAME in OPENROUTER_MODELS:
     # Create a set to remove duplicates, then convert back to list
     OPENROUTER_MODELS = list(dict.fromkeys(OPENROUTER_MODELS))

# --- API Call Functions ---

def call_openrouter_api(model_name, prompt, api_key, is_evaluation=False):
    """Calls the OpenRouter API for generation or evaluation."""
    log_prefix = "Evaluating with" if is_evaluation else "Querying"
    st.write(f"{log_prefix} OpenRouter: {model_name}...")

    messages = [{"role": "user", "content": prompt}]
    max_tokens = 50 if is_evaluation else 1024 # Shorter for eval expected output
    temperature = 0.1 if is_evaluation else 0.7 # Low temp for consistent eval

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }),
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not parse response.")
        return content.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling OpenRouter model {model_name}: {e}")
        return f"Error: API call failed - {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred with model {model_name}: {e}")
        return f"Error: Unexpected error - {e}"

def call_ollama_api(prompt):
    """Calls the local Ollama API."""
    st.write(f"Querying Ollama: {OLLAMA_MODEL_TAG}...")
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"model": OLLAMA_MODEL_TAG, "prompt": prompt, "stream": False})
    try:
        response = requests.post(url=OLLAMA_API_URL, headers=headers, data=payload, timeout=180)
        response.raise_for_status()
        # Handle potential non-JSON responses or different structures if Ollama version changes
        try:
            data = response.json()
            content = data.get("response", f"Error: 'response' key missing in Ollama JSON. Response: {response.text[:200]}")
        except json.JSONDecodeError:
             content = f"Error: Ollama returned non-JSON response: {response.text[:200]}"
        return content.strip()
    except requests.exceptions.ConnectionError as e:
        st.error(f"Error connecting to Ollama at {OLLAMA_API_URL}. Is Ollama running? Error: {e}")
        return f"Error: Connection failed - {e}"
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama model {OLLAMA_MODEL_TAG}: {e}")
        return f"Error: API call failed - {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred with Ollama model {OLLAMA_MODEL_TAG}: {e}")
        return f"Error: Unexpected error - {e}"


# --- MODIFIED LLM-based Evaluation Function ---
# --- MODIFIED LLM-based Evaluation Function ---
def evaluate_response_llm(original_prompt, candidate_response, reference_answer, api_key):
    """Evaluates a candidate response against a user-provided reference answer using an LLM judge."""

    # Check prerequisites
    if not api_key:
        st.error("OpenRouter API Key is required for LLM-based evaluation.")
        return "Error: API Key Missing"
    if not reference_answer:
        st.warning("No reference answer provided. Skipping LLM evaluation.")
        return "N/A (No Ref. Answer)"
    if candidate_response.startswith("Error:"): # Don't evaluate if generation failed
        return "N/A (Generation Error)"

    # --- !!! REVISED Evaluation Prompt for Stricter Judgement !!! ---
    # This prompt emphasizes accuracy, unambiguity, and penalizes misleading info.
    evaluation_prompt = f"""You are a meticulous and impartial judge evaluating an AI model's response based on a known correct reference answer.

Original User Question:
\"\"\"
{original_prompt}
\"\"\"

Reference Answer (Considered Ground Truth):
\"\"\"
{reference_answer}
\"\"\"

Candidate Response (Generated by AI model):
\"\"\"
{candidate_response}
\"\"\"

**Your Task:** Analyze the **Candidate Response** strictly against the **Reference Answer**. The goal is to determine if the candidate provides the *correct and essential information* from the reference, without adding misleading or incorrect details relevant to the core question.

**Evaluation Criteria:**
1.  **Accuracy:** Does the candidate accurately state the key information from the reference? (e.g., If the reference says 'H7', does the candidate correctly identify 'H7'?)
2.  **Clarity & Unambiguity:** Is the correct information presented clearly? Is it mixed with incorrect suggestions that make the answer confusing or potentially lead the user astray? (e.g., Listing 'H7' alongside incorrect bulb types like 'H11' or '9004' *as equally valid options for the primary headlight* makes the answer ambiguous and INCORRECT, even if 'H7' is present).
3.  **Completeness (Essential Info):** Does the candidate include the *most critical* piece of information from the reference needed to answer the user's core question? (e.g., Identifying the bulb type).
4.  **Focus:** Does the candidate directly address the core question, or does it primarily offer generic advice while being vague about the specific answer contained in the reference? A response that mentions the correct answer buried in generic advice and incorrect options is less valuable and potentially INCORRECT.

A response is **Correct** ONLY IF it accurately reflects the essential information from the reference answer clearly and without significant misleading additions or ambiguity regarding the core point.

A response is **Incorrect** IF:
- It fails to provide the essential information from the reference.
- It provides the essential information but mixes it ambiguously with incorrect information (like wrong bulb types suggested as equals).
- It is overly vague and evasive, prioritizing generic advice over the specific answer known from the reference.

Based on this **strict evaluation**, comparing the **Candidate Response** to the **Reference Answer**, is the candidate response correct?

Respond with only the single word 'Correct' or 'Incorrect'. Do not provide any explanation or justification.
"""
    # --- End of revised evaluation prompt ---

    eval_response = call_openrouter_api(
        model_name=EVALUATION_MODEL_NAME,
        prompt=evaluation_prompt,
        api_key=api_key,
        is_evaluation=True # Signal evaluation call
    )

    # Parse the evaluation response (keep previous parsing logic)
    if eval_response.startswith("Error:"):
        return "Eval Error"

    cleaned_eval_response = eval_response.strip().lower()

    if cleaned_eval_response == "correct":
        return "Correct"
    elif cleaned_eval_response == "incorrect":
        return "Incorrect"
    # Handle cases where the judge LLM might not follow instructions perfectly
    elif "correct" in cleaned_eval_response:
        st.warning(f"Evaluation model ({EVALUATION_MODEL_NAME}) returned: '{eval_response}'. Interpreting as 'Correct' due to keyword presence.")
        return "Correct (imprecise)"
    elif "incorrect" in cleaned_eval_response:
        st.warning(f"Evaluation model ({EVALUATION_MODEL_NAME}) returned: '{eval_response}'. Interpreting as 'Incorrect' due to keyword presence.")
        return "Incorrect (imprecise)"
    else:
        st.warning(f"Unexpected evaluation response from {EVALUATION_MODEL_NAME}: '{eval_response}'. Marking as Undetermined.")
        return "Undetermined"

# --- Rest of your llm_comparator.py script remains the same ---
# (Make sure to replace the old evaluate_response_llm function with this revised one)

    eval_response = call_openrouter_api(
        model_name=EVALUATION_MODEL_NAME,
        prompt=evaluation_prompt,
        api_key=api_key,
        is_evaluation=True # Signal evaluation call for specific parameters
    )

    # Parse the evaluation response
    if eval_response.startswith("Error:"):
        return "Eval Error"

    cleaned_eval_response = eval_response.strip().lower()

    if cleaned_eval_response == "correct":
        return "Correct"
    elif cleaned_eval_response == "incorrect":
        return "Incorrect"
    # Handle cases where the judge LLM might not follow instructions perfectly
    elif "correct" in cleaned_eval_response:
        st.warning(f"Evaluation model ({EVALUATION_MODEL_NAME}) returned: '{eval_response}'. Interpreting as 'Correct' due to keyword presence.")
        return "Correct (imprecise)"
    elif "incorrect" in cleaned_eval_response:
        st.warning(f"Evaluation model ({EVALUATION_MODEL_NAME}) returned: '{eval_response}'. Interpreting as 'Incorrect' due to keyword presence.")
        return "Incorrect (imprecise)"
    else:
        st.warning(f"Unexpected evaluation response from {EVALUATION_MODEL_NAME}: '{eval_response}'. Marking as Undetermined.")
        return "Undetermined"

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("LLM Comparison Tool")

# --- Sidebar for Configuration ---
st.sidebar.header("API Key & Config")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key (Required)", type="password",
    help=f"Required for OpenRouter models AND for evaluation using {EVALUATION_MODEL_NAME}."
)
st.sidebar.info(f"Local Ollama Model: `{OLLAMA_MODEL_TAG}` (No Key Needed)")
st.sidebar.info(f"Evaluation Model: `{EVALUATION_MODEL_NAME}` (Uses OpenRouter Key)")

st.sidebar.header("Model Selection")
all_model_options = [FINE_TUNED_MODEL_UI_NAME] + OPENROUTER_MODELS
selected_models = st.sidebar.multiselect(
    "Select Models to Compare", options=all_model_options,
    default=[FINE_TUNED_MODEL_UI_NAME, "meta-llama/llama-3.1-8b-instruct", "google/gemini-flash-1.5"]
)

# --- Main Area ---
st.header("Inputs")
default_prompt = "What model of head light should I use as a replacement for my Chrysler Crossfire?"
user_prompt = st.text_area("1. Enter the prompt to send to the models:", value=default_prompt, height=100)

# --- NEW: Input for Reference Answer ---
st.subheader("2. Provide Reference Answer for Evaluation")
reference_answer = st.text_area(
    "Enter the 'correct' or 'ideal' answer for the prompt above. The evaluation LLM will compare generated responses against this.",
    value="For a Chrysler Crossfire, the standard low beam headlight replacement bulb is an H7.", # Example reference
    height=100,
    help="Optional, but required for LLM-based evaluation. Leave blank to skip evaluation."
)

if st.button("Run Comparison & Evaluation"):
    # Basic Input Checks
    if not user_prompt:
        st.warning("Please enter a prompt.")
    elif not selected_models:
        st.warning("Please select at least one model.")
    elif not openrouter_api_key:
         st.error("OpenRouter API Key is required in the sidebar for evaluation.")
    # Removed check for reference_answer here, handled in evaluation step

    else:
        st.header("Results")
        results = []
        all_responses = {} # Store generated responses

        with st.spinner("Querying models and performing evaluation... Please wait."):
            # Step 1: Generate responses
            generation_responses = {}
            generation_latencies = {}
            st.subheader("1. Generating Responses...")
            progress_bar_gen = st.progress(0, text="Generating...")
            for i, model_ui_name in enumerate(selected_models):
                progress_text = f"Generating response from {model_ui_name}..."
                progress_bar_gen.progress((i + 1) / len(selected_models), text=progress_text)
                response_content = None
                start_time = time.time()

                if model_ui_name == FINE_TUNED_MODEL_UI_NAME:
                    response_content = call_ollama_api(user_prompt)
                elif model_ui_name in OPENROUTER_MODELS:
                    response_content = call_openrouter_api(model_ui_name, user_prompt, openrouter_api_key, is_evaluation=False)
                else:
                    st.error(f"Model {model_ui_name} not recognized.")
                    response_content = "Error: Model not recognized."

                end_time = time.time()
                latency = end_time - start_time
                generation_responses[model_ui_name] = response_content
                generation_latencies[model_ui_name] = latency

            # Step 2: Evaluate responses (if reference answer is provided)
            evaluation_results = {}
            evaluation_latencies = {}
            st.subheader("2. Evaluating Responses...")
            if reference_answer: # Only run evaluation if reference is given
                progress_bar_eval = st.progress(0, text="Evaluating...")
                for i, model_ui_name in enumerate(selected_models):
                    progress_text = f"Evaluating response from {model_ui_name} using {EVALUATION_MODEL_NAME}..."
                    progress_bar_eval.progress((i + 1) / len(selected_models), text=progress_text)
                    candidate_response = generation_responses[model_ui_name]
                    start_eval_time = time.time()
                    # *** Pass reference_answer to the evaluation function ***
                    eval_result = evaluate_response_llm(user_prompt, candidate_response, reference_answer, openrouter_api_key)
                    end_eval_time = time.time()
                    evaluation_results[model_ui_name] = eval_result
                    evaluation_latencies[model_ui_name] = end_eval_time - start_eval_time
            else:
                st.info("No reference answer provided, skipping LLM evaluation step.")
                # Fill results with N/A if no evaluation was performed
                for model_ui_name in selected_models:
                    evaluation_results[model_ui_name] = "N/A (No Ref. Answer)"
                    evaluation_latencies[model_ui_name] = 0.0

            # Step 3: Combine results
            st.subheader("3. Compiling Results...")
            for model_ui_name in selected_models:
                 results.append({
                    "Model": model_ui_name,
                    "Response": generation_responses[model_ui_name],
                    # Updated column name for clarity
                    f"Eval vs Ref. ({EVALUATION_MODEL_NAME})": evaluation_results[model_ui_name],
                    "Generation Latency (s)": f"{generation_latencies[model_ui_name]:.2f}",
                    "Evaluation Latency (s)": f"{evaluation_latencies.get(model_ui_name, 0.0):.2f}", # Use .get for safety
                 })
                 # Store for side-by-side display
                 all_responses[model_ui_name] = generation_responses[model_ui_name]

        st.success("Generation and Evaluation Complete!")

        # --- Display Comparison Table ---
        st.header("Comparison Summary Table")
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, height=max(300, len(results)*40)) # Adjust height dynamically
        else:
            st.info("No results to display.")

        # --- Display Full Responses ---
        st.header("Full Generated Responses")
        if all_responses:
            num_models = len(all_responses)
            cols = st.columns(min(num_models, 3)) # Max 3 columns
            col_index = 0
            model_names = list(all_responses.keys())
            for i in range(num_models):
                model_name = model_names[i]
                response_text = all_responses[model_name]
                current_col = cols[col_index % len(cols)]
                with current_col:
                    st.markdown(f"**{model_name}**")
                    st.text_area(label=f"response_{model_name}",
                                 value=response_text, height=400,
                                 key=f"textarea_{model_name}_{i}", disabled=True) # Unique key
                col_index += 1
        else:
            st.info("No responses were generated.")


# --- How to Use Section ---
st.markdown("---")
st.header("How to Use")
st.markdown(f"""
1.  **Ensure Ollama is Running:** (If using local model) Make sure `{OLLAMA_MODEL_TAG}` is pulled and Ollama is running.
2.  **Enter OpenRouter API Key:** Provide your OpenRouter API key in the sidebar (Required for evaluation using `{EVALUATION_MODEL_NAME}`).
3.  **Select Models:** Choose the models to compare.
4.  **Enter Prompt:** Input the question/task for the models.
5.  **Provide Reference Answer:** In the second text box, enter the answer you consider correct or ideal for the prompt. This is crucial for the LLM evaluation step. If you leave this blank, evaluation will be skipped.
6.  **Run Comparison & Evaluation:** Click the button.
7.  **View Results:**
    *   **Summary Table:** Shows models, responses, the evaluation result ('Correct'/'Incorrect' based on comparison to *your* reference answer), and latencies.
    *   **Full Generated Responses:** Displays the complete text generated by each model.
""")
st.markdown(f"""
**Note on Evaluation:** The quality of the 'Correct'/'Incorrect' judgment depends heavily on the clarity of your **Reference Answer** and the capability of the chosen evaluation model (`{EVALUATION_MODEL_NAME}`). The evaluation prompt instructs the judge LLM to check for consistency and essential information match.
""")
