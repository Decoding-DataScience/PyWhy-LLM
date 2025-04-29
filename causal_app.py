import streamlit as st
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm.suggesters.identification_suggester import IdentificationSuggester
from pywhyllm.suggesters.validation_suggester import ValidationSuggester
from pywhyllm import RelationshipStrategy
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
else:
    st.title("PyWhy-LLM Causal Analysis Assistant")
    st.subheader("Leveraging LLMs for Causal Insights")

    llm_model = st.sidebar.selectbox("Choose LLM Model", ["gpt-4"])  # Currently only gpt-4 is shown in examples

    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Step", ["Model Suggestion", "Identification Suggestion", "Validation Suggestion"]
    )

    all_factors_str = st.text_area("Enter all relevant factors (comma-separated):", "smoking, lung cancer, exercise habits, air pollution exposure")
    all_factors = [factor.strip() for factor in all_factors_str.split(',')]

    treatment = st.text_input("Enter the treatment variable:")
    outcome = st.text_input("Enter the outcome variable:")

    # Initialize session state for domain_expertises if it doesn't exist
    if 'domain_expertises' not in st.session_state:
        st.session_state.domain_expertises = None

    if analysis_type == "Model Suggestion":
        modeler = ModelSuggester(llm_model)

        if st.button("Suggest Domain Expertises"):
            if all_factors:
                st.session_state.domain_expertises = modeler.suggest_domain_expertises(all_factors)
                st.subheader("Suggested Domain Expertises:")
                st.write(st.session_state.domain_expertises)
            else:
                st.warning("Please enter the relevant factors.")

        if st.button("Suggest Potential Confounders"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_confounders = modeler.suggest_confounders(treatment, outcome, all_factors, st.session_state.domain_expertises)
                st.subheader("Suggested Potential Confounders:")
                st.write(suggested_confounders)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        if st.button("Suggest Pair-wise Relationships (DAG)"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_dag = modeler.suggest_relationships(
                    treatment, outcome, all_factors, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                )
                st.subheader("Suggested Pair-wise Relationships (Potential DAG Edges):")
                st.write(suggested_dag)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

    elif analysis_type == "Identification Suggestion":
        identifier = IdentificationSuggester(llm_model)

        if st.button("Suggest Backdoor Set"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_backdoor = identifier.suggest_backdoor(treatment, outcome, all_factors, st.session_state.domain_expertises)
                st.subheader("Suggested Backdoor Set:")
                st.write(suggested_backdoor)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        if st.button("Suggest Mediator Set"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_mediators = identifier.suggest_mediators(treatment, outcome, all_factors, st.session_state.domain_expertises)
                st.subheader("Suggested Mediator Set:")
                st.write(suggested_mediators)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        if st.button("Suggest Instrumental Variables (IVs)"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_iv = identifier.suggest_ivs(treatment, outcome, all_factors, st.session_state.domain_expertises)
                st.subheader("Suggested Instrumental Variables (IVs):")
                st.write(suggested_iv)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

    elif analysis_type == "Validation Suggestion":
        validator = ValidationSuggester(llm_model)

        dag_str = st.text_area(
            "Enter the suggested DAG edges as a dictionary (e.g., {'smoking': ['lung cancer'], 'air pollution exposure': ['lung cancer']}):",
            "{}"
        )
        try:
            suggested_dag = eval(dag_str)
        except (SyntaxError, NameError):
            suggested_dag = {}
            st.warning("Please enter a valid Python dictionary for the DAG.")

        if st.button("Critique the DAG Edges"):
            if all_factors and suggested_dag and st.session_state.domain_expertises is not None:
                suggested_critiques_dag = validator.critique_graph(
                    all_factors, suggested_dag, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                )
                st.subheader("Critique of the DAG Edges:")
                st.write(suggested_critiques_dag)
            else:
                st.warning("Please ensure factors, DAG, and domain expertises are provided.")

        if st.button("Suggest Latent Confounders"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_latent_confounders = validator.suggest_latent_confounders(
                    treatment, outcome, all_factors, st.session_state.domain_expertises
                )
                st.subheader("Suggested Latent Confounders:")
                st.write(suggested_latent_confounders)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        if st.button("Suggest Negative Controls"):
            if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                suggested_negative_controls = validator.suggest_negative_controls(
                    treatment, outcome, all_factors, st.session_state.domain_expertises
                )
                st.subheader("Suggested Negative Controls:")
                st.write(suggested_negative_controls)
            else:
                st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")