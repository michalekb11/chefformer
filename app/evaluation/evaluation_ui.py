import streamlit as st
import json
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Chefformer Evaluation", layout="wide")

# Path relative to the project root where the script is expected to run
RESULTS_PATH = "./logs/pretrain/evaluation/eval_results.json"

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    try:
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

data = load_results()

if not data:
    st.title("👨‍🍳 Chefformer Evaluation")
    st.error(f"Evaluation results not found at `{RESULTS_PATH}`.")
    st.info("Please run the evaluation script first: `python src/evaluation/evaluate.py --checkpoint_dir ... --steps ...`")
    st.stop()

steps = sorted(data.keys(), key=int)

st.title("👨‍🍳 Chefformer: The Learning Journey")

tab1, tab2 = st.tabs(["📈 Quantitative Metrics", "📖 Qualitative Growth"])

with tab1:
    st.header("Training Progress Metrics")
    metrics_list = []
    for step in steps:
        m = data[step]["metrics"]
        m["Step"] = int(step)
        metrics_list.append(m)
    
    df = pd.DataFrame(metrics_list)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_div = px.line(df, x="Step", y="avg_diversity", title="3-Gram Diversity (Higher is Better)")
        st.plotly_chart(fig_div, use_container_width=True)
    with col2:
        fig_bleu = px.line(df, x="Step", y="self_bleu", title="Self-BLEU (Lower is more Diverse)")
        st.plotly_chart(fig_bleu, use_container_width=True)
    with col3:
        fig_val = px.line(df, x="Step", y="avg_lexical_validity", title="Lexical Validity (Higher is Better)")
        st.plotly_chart(fig_val, use_container_width=True)

with tab2:
    st.header("Prompt Comparison Across Time")
    
    # Safely get prompt options
    first_step_prompts = data[steps[0]].get("prompts", [])
    if not first_step_prompts:
        st.warning("No prompt data found in the results.")
        st.stop()

    prompt_options = [p["input"] for p in first_step_prompts]
    selected_prompt = st.selectbox("Select a Prompt to Track:", prompt_options)
    
    for step in steps:
        with st.expander(f"Step {step}"):
            step_prompts = data[step].get("prompts", [])
            output_data = next((p for p in step_prompts if p["input"] == selected_prompt), None)
            prompt_idx = prompt_options.index(selected_prompt)
            
            if output_data:
                # Adding prompt_idx to the key forces Streamlit to refresh the widget content when prompt changes
                st.text_area(label="Model Output", value=output_data["output"], height=300, key=f"text_{step}_{prompt_idx}")
                st.caption(f"Diversity Score: {output_data.get('diversity', 0.0):.4f}")
            else:
                st.write("No output for this prompt at this step.")
