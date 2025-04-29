import streamlit as st
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm.suggesters.identification_suggester import IdentificationSuggester
from pywhyllm.suggesters.validation_suggester import ValidationSuggester
from pywhyllm import RelationshipStrategy
import os
import json
from dotenv import load_dotenv

def convert_tuples_to_lists(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_tuples_to_lists(value) for key, value in obj.items()}
    else:
        return obj

def format_relationship_output(relationships):
    """Format relationships into readable text."""
    if not relationships:
        return "_No relationships found._"
    
    formatted_output = []
    for rel in relationships:
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            # Handle relationship tuples/lists with scores
            if len(rel) == 3 and isinstance(rel[2], (int, float)):
                formatted_output.append(f"• {rel[0]} → {rel[1]} (score: {rel[2]})")
            else:
                formatted_output.append(f"• {rel[0]} → {rel[1]}")
        elif isinstance(rel, dict):
            # Handle dictionary format
            for source, targets in rel.items():
                if isinstance(targets, list):
                    for target in targets:
                        formatted_output.append(f"• {source} → {target}")
                else:
                    formatted_output.append(f"• {source} → {targets}")
        else:
            # Fallback for other formats
            formatted_output.append(f"• {rel}")
    
    return "\n".join(formatted_output)

def format_domain_expertises(expertises):
    """Format domain expertises into readable text."""
    if not expertises:
        return "_No domain expertises found._"
    
    if isinstance(expertises, dict):
        formatted_output = []
        for domain, expertise in expertises.items():
            formatted_output.append(f"### {domain}")
            if isinstance(expertise, list):
                formatted_output.extend([f"- {item}" for item in expertise])
            else:
                formatted_output.append(f"- {expertise}")
            formatted_output.append("")  # Add blank line between domains
        return "\n".join(formatted_output)
    elif isinstance(expertises, list):
        return "\n".join([f"- {expertise}" for expertise in expertises])
    else:
        return str(expertises)

def format_critiques(critiques):
    """Format critiques into readable text."""
    if not critiques:
        return "_No critiques found._"
    
    if isinstance(critiques, dict):
        formatted_output = []
        for category, items in critiques.items():
            formatted_output.append(f"### {category}")
            if isinstance(items, list):
                formatted_output.extend([f"- {item}" for item in items])
            else:
                formatted_output.append(f"- {items}")
            formatted_output.append("")  # Add blank line between categories
        return "\n".join(formatted_output)
    elif isinstance(critiques, list):
        return "\n".join([f"- {critique}" for critique in critiques])
    else:
        return str(critiques)

def format_variables(variables):
    """Format variables into readable text."""
    if not variables:
        return "_No variables found._"
    
    if isinstance(variables, list):
        return "\n".join([f"- {var}" for var in variables])
    elif isinstance(variables, dict):
        formatted_output = []
        for category, vars in variables.items():
            formatted_output.append(f"### {category}")
            if isinstance(vars, list):
                formatted_output.extend([f"- {var}" for var in vars])
            else:
                formatted_output.append(f"- {vars}")
            formatted_output.append("")
        return "\n".join(formatted_output)
    else:
        return str(variables)

def validate_dag_input(dag_str):
    """Validate DAG input and return tuple of (is_valid, dag_dict, error_message)."""
    if not dag_str or dag_str.strip() == '{}':
        return False, {}, "Please enter a DAG structure."
    
    try:
        # First try to parse as JSON
        dag_dict = json.loads(dag_str)
        if not isinstance(dag_dict, dict):
            return False, {}, "Input must be a dictionary/object."
        
        # Validate structure
        for source, targets in dag_dict.items():
            if not isinstance(source, str):
                return False, {}, f"Key '{source}' must be a string."
            if not isinstance(targets, list):
                if isinstance(targets, str):
                    # Convert single string to list
                    dag_dict[source] = [targets]
                else:
                    return False, {}, f"Value for '{source}' must be a list of strings or a single string."
            else:
                for target in targets:
                    if not isinstance(target, str):
                        return False, {}, f"Target '{target}' in list for '{source}' must be a string."
        
        return True, dag_dict, ""
    except json.JSONDecodeError:
        try:
            # Try to evaluate as Python dict if JSON fails
            dag_dict = eval(dag_str)
            if not isinstance(dag_dict, dict):
                return False, {}, "Input must be a dictionary."
            
            # Convert to proper format
            formatted_dict = {}
            for source, targets in dag_dict.items():
                if isinstance(targets, str):
                    formatted_dict[source] = [targets]
                elif isinstance(targets, (list, tuple)):
                    formatted_dict[source] = list(targets)
                else:
                    return False, {}, f"Value for '{source}' must be a list of strings or a single string."
            
            return True, formatted_dict, ""
        except:
            return False, {}, "Invalid input format. Please use valid JSON or Python dictionary format."

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
else:
    # Create two columns for the layout
    guide_col, main_col = st.columns([1, 2])

    with guide_col:
        st.markdown("""
        # Application Guide
        
        ## What is PyWhy-LLM?
        PyWhy-LLM is an innovative tool that combines Large Language Models (LLMs) with causal analysis. It helps researchers and analysts:
        - Identify potential causal relationships
        - Suggest confounding variables
        - Validate causal assumptions
        - Build and critique DAGs (Directed Acyclic Graphs)
        
        ## How to Use
        1. Choose your LLM model (currently GPT-4)
        2. Select an analysis step
        3. Enter your variables and factors
        4. Follow the step-by-step process
        
        ## Analysis Steps
        
        ### 1. Model Suggestion
        - Start here to build your causal model
        - Identify domain expertise needed
        - Discover potential confounders
        - Build initial relationships
        
        ### 2. Identification Suggestion
        - Find backdoor adjustment sets
        - Identify mediator variables
        - Discover instrumental variables
        
        ### 3. Validation Suggestion
        - Validate your DAG structure
        - Find latent confounders
        - Identify negative controls
        
        ## Tips
        - Be specific with your variables
        - Use clear, consistent naming
        - Consider all relevant factors
        - Start with simple relationships
        """)

    with main_col:
        st.title("PyWhy-LLM Causal Analysis Assistant")
        st.subheader("Leveraging LLMs for Causal Insights")

        llm_model = st.sidebar.selectbox("Choose LLM Model", ["gpt-4"])

        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Step", ["Model Suggestion", "Identification Suggestion", "Validation Suggestion"]
        )

        # Add help text for factors with examples
        st.markdown("""
        ### Enter Variables and Factors
        Provide all relevant variables that might be involved in your causal analysis.
        """)
        
        factors_help = """
        Examples:
        - Medical: "smoking, lung cancer, exercise habits, air pollution exposure"
        - Education: "study hours, test scores, sleep quality, stress levels"
        - Economics: "interest rates, inflation, unemployment, gdp growth"
        """
        
        all_factors_str = st.text_area(
            "Enter all relevant factors (comma-separated):", 
            "smoking, lung cancer, exercise habits, air pollution exposure",
            help=factors_help
        )
        all_factors = [factor.strip() for factor in all_factors_str.split(',')]

        # Add help text for treatment with examples
        treatment_help = """
        Examples:
        - Medical: "smoking" or "exercise_program"
        - Education: "study_hours" or "tutoring_program"
        - Economics: "interest_rate_change" or "tax_policy"
        """
        treatment = st.text_input(
            "Enter the treatment variable:", 
            help=treatment_help
        )

        # Add help text for outcome with examples
        outcome_help = """
        Examples:
        - Medical: "lung_cancer" or "blood_pressure"
        - Education: "test_score" or "graduation_rate"
        - Economics: "gdp_growth" or "unemployment_rate"
        """
        outcome = st.text_input(
            "Enter the outcome variable:",
            help=outcome_help
        )

        # Initialize session state for domain_expertises if it doesn't exist
        if 'domain_expertises' not in st.session_state:
            st.session_state.domain_expertises = None

        if analysis_type == "Model Suggestion":
            st.markdown("""
            ### Model Suggestion Step
            This step helps you build the initial causal model by:
            1. Identifying required domain expertise
            2. Suggesting potential confounding variables
            3. Building pair-wise relationships between variables
            """)
            
            modeler = ModelSuggester(llm_model)

            if st.button("Suggest Domain Expertises"):
                if all_factors:
                    st.session_state.domain_expertises = modeler.suggest_domain_expertises(all_factors)
                    st.subheader("Suggested Domain Expertises:")
                    formatted_expertises = format_domain_expertises(st.session_state.domain_expertises)
                    st.markdown(formatted_expertises)
                else:
                    st.warning("Please enter the relevant factors.")

            if st.button("Suggest Potential Confounders"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_confounders = modeler.suggest_confounders(treatment, outcome, all_factors, st.session_state.domain_expertises)
                    st.subheader("Suggested Potential Confounders:")
                    formatted_confounders = format_variables(convert_tuples_to_lists(suggested_confounders))
                    st.markdown(formatted_confounders)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

            if st.button("Suggest Pair-wise Relationships (DAG)"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_dag = modeler.suggest_relationships(
                        treatment, outcome, all_factors, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                    )
                    st.subheader("Suggested Pair-wise Relationships (Potential DAG Edges):")
                    formatted_dag = format_relationship_output(convert_tuples_to_lists(suggested_dag))
                    st.markdown(formatted_dag)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        elif analysis_type == "Identification Suggestion":
            st.markdown("""
            ### Identification Suggestion Step
            This step helps you identify:
            1. Backdoor adjustment sets for causal estimation
            2. Potential mediator variables
            3. Possible instrumental variables
            """)
            
            identifier = IdentificationSuggester(llm_model)

            if st.button("Suggest Backdoor Set"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_backdoor = identifier.suggest_backdoor(treatment, outcome, all_factors, st.session_state.domain_expertises)
                    st.subheader("Suggested Backdoor Set:")
                    formatted_backdoor = format_variables(convert_tuples_to_lists(suggested_backdoor))
                    st.markdown(formatted_backdoor)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

            if st.button("Suggest Mediator Set"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_mediators = identifier.suggest_mediators(treatment, outcome, all_factors, st.session_state.domain_expertises)
                    st.subheader("Suggested Mediator Set:")
                    formatted_mediators = format_relationship_output(convert_tuples_to_lists(suggested_mediators))
                    st.markdown(formatted_mediators)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

            if st.button("Suggest Instrumental Variables (IVs)"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_iv = identifier.suggest_ivs(treatment, outcome, all_factors, st.session_state.domain_expertises)
                    st.subheader("Suggested Instrumental Variables (IVs):")
                    formatted_iv = format_variables(convert_tuples_to_lists(suggested_iv))
                    st.markdown(formatted_iv)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

        elif analysis_type == "Validation Suggestion":
            st.markdown("""
            ### Validation Suggestion Step
            This step helps you:
            1. Validate your DAG structure
            2. Identify potential latent confounders
            3. Find suitable negative controls
            """)
            
            validator = ValidationSuggester(llm_model)

            # Example DAG for the current factors
            example_dag = {
                "smoking": ["lung cancer"],
                "air pollution exposure": ["lung cancer"],
                "exercise habits": ["lung cancer"]
            }
            
            st.markdown("""
            ### DAG Input Guide
            Enter your DAG structure using either format:
            1. JSON: `{"source": ["target1", "target2"], ...}`
            2. Python dict: `{'source': ['target1', 'target2'], ...}`
            
            Example for current factors:
            ```python
            {
                "smoking": ["lung cancer"],
                "air pollution exposure": ["lung cancer"],
                "exercise habits": ["lung cancer"]
            }
            ```
            """)

            dag_str = st.text_area(
                "Enter the suggested DAG edges:",
                "{}"
            )

            is_valid, suggested_dag, error_msg = validate_dag_input(dag_str)
            if not is_valid:
                st.error(error_msg)
            elif suggested_dag:  # If valid and not empty
                st.success("Valid DAG structure")
                st.markdown("### Current DAG Structure:")
                formatted_dag = format_relationship_output(suggested_dag)
                st.markdown(formatted_dag)

            if st.button("Critique the DAG Edges"):
                if all_factors and suggested_dag and st.session_state.domain_expertises is not None:
                    suggested_critiques_dag = validator.critique_graph(
                        all_factors, suggested_dag, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                    )
                    st.subheader("Critique of the DAG Edges:")
                    formatted_critiques = format_critiques(convert_tuples_to_lists(suggested_critiques_dag))
                    st.markdown(formatted_critiques)
                else:
                    st.warning("Please ensure factors, DAG, and domain expertises are provided.")

            if st.button("Suggest Latent Confounders"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_latent_confounders = validator.suggest_latent_confounders(
                        treatment, outcome, all_factors, st.session_state.domain_expertises
                    )
                    st.subheader("Suggested Latent Confounders:")
                    formatted_confounders = format_variables(convert_tuples_to_lists(suggested_latent_confounders))
                    st.markdown(formatted_confounders)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

            if st.button("Suggest Negative Controls"):
                if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                    suggested_negative_controls = validator.suggest_negative_controls(
                        treatment, outcome, all_factors, st.session_state.domain_expertises
                    )
                    st.subheader("Suggested Negative Controls:")
                    formatted_controls = format_variables(convert_tuples_to_lists(suggested_negative_controls))
                    st.markdown(formatted_controls)
                else:
                    st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")