import streamlit as st
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm.suggesters.identification_suggester import IdentificationSuggester
from pywhyllm.suggesters.validation_suggester import ValidationSuggester
from pywhyllm import RelationshipStrategy
import os
import json
from dotenv import load_dotenv

# Set page config
st.set_page_config(
    page_title="PyWhy-LLM Causal Analysis Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        color: #1E88E5;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    .section-header {
        color: #2E7D32;
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 10px;
    }
    .subsection-header {
        color: #1565C0;
        font-size: 20px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .step-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    """Format relationships into readable text with explanations."""
    if not relationships:
        return "_No relationships found._"
    
    def format_single_item(item):
        return str(item).replace('_', ' ').title()
    
    formatted_output = []
    for rel in relationships:
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            if len(rel) == 3 and isinstance(rel[2], (int, float)):
                formatted_output.extend([
                    f"• {format_single_item(rel[0])} affects {format_single_item(rel[1])}",
                    f"  Confidence Score: {rel[2]:.2f}",
                    f"  💡 This relationship suggests that changes in {format_single_item(rel[0]).lower()} may lead to changes in {format_single_item(rel[1]).lower()}.",
                    ""  # Add blank line
                ])
            else:
                formatted_output.extend([
                    f"• {format_single_item(rel[0])} affects {format_single_item(rel[1])}",
                    f"  💡 This indicates a potential causal link from {format_single_item(rel[0]).lower()} to {format_single_item(rel[1]).lower()}.",
                    ""  # Add blank line
                ])
        elif isinstance(rel, dict):
            for source, targets in rel.items():
                if isinstance(targets, (list, tuple)):
                    for target in targets:
                        formatted_output.extend([
                            f"• {format_single_item(source)} influences {format_single_item(target)}",
                            f"  💡 This suggests that {format_single_item(source).lower()} may have an effect on {format_single_item(target).lower()}.",
                            ""  # Add blank line
                        ])
                else:
                    formatted_output.extend([
                        f"• {format_single_item(source)} influences {format_single_item(targets)}",
                        f"  💡 This indicates a potential relationship where {format_single_item(source).lower()} affects {format_single_item(targets).lower()}.",
                        ""  # Add blank line
                    ])
    
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
    """Format critiques into readable text with explanations."""
    if not critiques:
        return "_No critiques found._"
    
    def format_single_critique(critique):
        return str(critique).replace('_', ' ').title()
    
    if isinstance(critiques, dict):
        formatted_output = []
        for category, items in critiques.items():
            formatted_output.append(f"### {str(category).replace('_', ' ').title()}")
            formatted_output.append("")  # Add spacing after category
            
            if isinstance(items, list):
                for item in items:
                    formatted_output.extend([
                        f"• {format_single_critique(item)}",
                        f"  💡 {generate_critique_explanation(category, item)}",
                        ""  # Add blank line
                    ])
            else:
                formatted_output.extend([
                    f"• {format_single_critique(items)}",
                    f"  💡 {generate_critique_explanation(category, items)}",
                    ""  # Add blank line
                ])
            
            formatted_output.append("")  # Add extra spacing between categories
        return "\n".join(formatted_output)
    elif isinstance(critiques, list):
        formatted_output = []
        for critique in critiques:
            formatted_output.extend([
                f"• {format_single_critique(critique)}",
                f"  💡 {generate_critique_explanation('general', critique)}",
                ""  # Add blank line
            ])
        return "\n".join(formatted_output)
    else:
        return f"• {format_single_critique(critiques)}\n  💡 {generate_critique_explanation('general', critiques)}"

def generate_critique_explanation(category, critique):
    """Generate human-readable explanations for critiques."""
    category = category.lower()
    critique = str(critique).lower()
    
    if 'missing' in category or 'missing' in critique:
        return "Consider adding this element to make your causal model more complete."
    elif 'invalid' in category or 'invalid' in critique:
        return "This relationship may need to be reconsidered or removed from your model."
    elif 'suggestion' in category:
        return "This is a recommended addition to improve your causal model."
    elif 'warning' in category:
        return "Pay attention to this aspect as it might affect the validity of your analysis."
    else:
        return "Consider this point to improve your causal analysis."

def format_variables(variables):
    """Format variables into readable text with explanations."""
    if not variables:
        return "_No variables identified._"
    
    def format_single_var(var):
        """Helper function to format a single variable."""
        if isinstance(var, (list, tuple)):
            return [str(v).replace('_', ' ').title() for v in var]
        return str(var).replace('_', ' ').title()
    
    if isinstance(variables, list):
        formatted_vars = []
        for var in variables:
            if isinstance(var, (list, tuple)):
                if len(var) == 3 and isinstance(var[2], (int, float)):
                    formatted_vars.append({
                        "relationship": f"• {format_single_var(var[0])} affects {format_single_var(var[1])}",
                        "confidence": f"  Confidence Score: {var[2]:.2f}",
                        "explanation": f"  💡 This indicates a potential causal relationship where {format_single_var(var[0]).lower()} may influence {format_single_var(var[1]).lower()}."
                    })
                elif len(var) == 2:
                    formatted_vars.append({
                        "relationship": f"• {format_single_var(var[0])} affects {format_single_var(var[1])}",
                        "explanation": f"  💡 This suggests a direct relationship between {format_single_var(var[0]).lower()} and {format_single_var(var[1]).lower()}."
                    })
                else:
                    formatted_vars.append({
                        "relationship": f"• {', '.join(map(format_single_var, var))}",
                        "explanation": "  💡 These variables are related in the causal model."
                    })
            else:
                formatted_vars.append({
                    "variable": f"• {format_single_var(var)}",
                    "explanation": f"  💡 This is a key variable in your causal analysis."
                })
        
        # Format the output with each item on a new line
        output = []
        for item in formatted_vars:
            if "relationship" in item:
                output.append(item["relationship"])
                if "confidence" in item:
                    output.append(item["confidence"])
                output.append(item["explanation"])
            else:
                output.append(item["variable"])
                output.append(item["explanation"])
            output.append("")  # Add blank line between items
        
        return "\n".join(output)
    
    elif isinstance(variables, dict):
        formatted_output = []
        for category, vars in variables.items():
            formatted_output.append(f"### {str(category).replace('_', ' ').title()}")
            formatted_output.append("")  # Add spacing after category header
            
            if isinstance(vars, (list, tuple)):
                for var in vars:
                    formatted_output.append(f"• {format_single_var(var)}")
                    formatted_output.append(f"  💡 This is a relevant factor in the {category.lower()} category.")
                    formatted_output.append("")  # Add spacing between items
            else:
                formatted_output.append(f"• {format_single_var(vars)}")
                formatted_output.append(f"  💡 This is a key factor in the {category.lower()} category.")
                formatted_output.append("")
            
            formatted_output.append("")  # Add extra spacing between categories
        return "\n".join(formatted_output)
    else:
        return f"• {format_single_var(variables)}\n  💡 This is an important factor in your analysis."

def validate_dag_input(dag_str):
    """Validate DAG input and return tuple of (is_valid, dag_dict, error_message)."""
    if not dag_str or dag_str.strip() == '{}':
        return False, {}, "Please enter a DAG structure."
    
    try:
        # First try to parse as JSON
        dag_dict = json.loads(dag_str)
        if not isinstance(dag_dict, dict):
            return False, {}, "❌ Input must be a dictionary/object."
        
        # Validate structure
        for source, targets in dag_dict.items():
            if not isinstance(source, str):
                return False, {}, f"❌ Key '{source}' must be a string."
            if not isinstance(targets, list):
                if isinstance(targets, str):
                    # Convert single string to list
                    dag_dict[source] = [targets]
                else:
                    return False, {}, f"❌ Value for '{source}' must be a list of strings or a single string."
            else:
                for target in targets:
                    if not isinstance(target, str):
                        return False, {}, f"❌ Target '{target}' in list for '{source}' must be a string."
        
        return True, dag_dict, "✅ Valid DAG structure"
    except json.JSONDecodeError:
        try:
            # Try to evaluate as Python dict if JSON fails
            dag_dict = eval(dag_str)
            if not isinstance(dag_dict, dict):
                return False, {}, "❌ Input must be a dictionary."
            
            # Convert to proper format
            formatted_dict = {}
            for source, targets in dag_dict.items():
                if isinstance(targets, str):
                    formatted_dict[source] = [targets]
                elif isinstance(targets, (list, tuple)):
                    formatted_dict[source] = list(targets)
                else:
                    return False, {}, f"❌ Value for '{source}' must be a list of strings or a single string."
            
            return True, formatted_dict, "✅ Valid DAG structure"
        except:
            return False, {}, "❌ Invalid input format. Please use valid JSON or Python dictionary format."

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
else:
    # Main title
    st.markdown('<p class="main-title">PyWhy-LLM Causal Analysis Assistant</p>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown('<p class="section-header">Welcome to PyWhy-LLM</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ What is PyWhy-LLM?", expanded=True):
        st.markdown("""
        <div class="info-box">
        PyWhy-LLM is an innovative tool that combines Large Language Models (LLMs) with causal analysis to help researchers and analysts:
        
        - 🔍 Identify potential causal relationships
        - 🎯 Suggest confounding variables
        - ✅ Validate causal assumptions
        - 📊 Build and critique DAGs (Directed Acyclic Graphs)
        </div>
        """, unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown('<p class="section-header">Quick Start Guide</p>', unsafe_allow_html=True)
    with st.expander("📚 How to Use PyWhy-LLM", expanded=True):
        st.markdown("""
        <div class="step-box">
        1. 🎯 Select your analysis step from the sidebar
        2. 🔄 Enter your variables and factors in the input fields
        3. 📊 Follow the step-by-step process for your chosen analysis
        4. 📋 Review and interpret the results
        </div>
        """, unsafe_allow_html=True)

    # Main Analysis Interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<p class="section-header">Analysis Configuration</p>', unsafe_allow_html=True)
        
        llm_model = st.selectbox("🤖 Choose LLM Model", ["gpt-4"])
        
        analysis_type = st.selectbox(
            "📊 Choose Analysis Step",
            ["Model Suggestion", "Identification Suggestion", "Validation Suggestion"]
        )
        
        st.markdown('<p class="subsection-header">Variables Input</p>', unsafe_allow_html=True)
        
        # Updated help text with more domain examples
        factors_help = """
        Examples by domain:
        🏥 Medical: "smoking, lung cancer, exercise habits, air pollution"
        📚 Education: "study hours, test scores, sleep quality, stress"
        💰 Economics: "interest rates, inflation, unemployment, gdp"
        🌍 Environmental: "co2 emissions, temperature, deforestation, rainfall"
        """
        
        # Default example for medical domain
        default_factors = "smoking, lung cancer, exercise habits, air pollution exposure"
        all_factors_str = st.text_area(
            "📝 Enter all relevant factors (comma-separated):", 
            value=default_factors,
            help=factors_help,
            placeholder="Enter factors like: smoking, lung cancer, exercise habits..."
        )
        all_factors = [factor.strip() for factor in all_factors_str.split(',')]

        # Default example for treatment variable
        treatment = st.text_input(
            "🎯 Enter the treatment variable:",
            value="smoking",
            help="The variable whose effect you want to study (e.g., smoking, study hours, exercise)",
            placeholder="Enter treatment variable (e.g., smoking)"
        )

        # Default example for outcome variable
        outcome = st.text_input(
            "🎯 Enter the outcome variable:",
            value="lung cancer",
            help="The variable you want to measure the effect on (e.g., lung cancer, test score, health)",
            placeholder="Enter outcome variable (e.g., lung cancer)"
        )

    with col2:
        st.markdown('<p class="section-header">Analysis Steps</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'domain_expertises' not in st.session_state:
            st.session_state.domain_expertises = None

        if analysis_type == "Model Suggestion":
            st.markdown("""
            <div class="info-box">
            <h3>🏗️ Model Suggestion Step</h3>
            Build your initial causal model through these steps:
            1. Identify required domain expertise
            2. Discover potential confounding variables
            3. Establish pair-wise relationships between variables
            </div>
            """, unsafe_allow_html=True)
            
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
                    # Improved formatting for confounders
                    if isinstance(suggested_confounders, list):
                        formatted_confounders = "\n".join([f"• {conf.replace('_', ' ').title()}" for conf in suggested_confounders])
                    else:
                        formatted_confounders = format_variables(convert_tuples_to_lists(suggested_confounders))
                    st.markdown(formatted_confounders)
                    
                    # Add explanation box
                    st.info("💡 These confounding variables might affect both your treatment and outcome variables. Consider controlling for them in your analysis.")
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
            <div class="info-box">
            <h3>🔍 Identification Suggestion Step</h3>
            Identify key components for causal estimation:
            1. Find backdoor adjustment sets
            2. Discover potential mediator variables
            3. Locate possible instrumental variables
            </div>
            """, unsafe_allow_html=True)
            
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
            <div class="info-box">
            <h3>✅ Validation Suggestion Step</h3>
            Validate and improve your causal model:
            1. Validate your DAG structure
            2. Identify potential latent confounders
            3. Find suitable negative controls
            </div>
            """, unsafe_allow_html=True)
            
            validator = ValidationSuggester(llm_model)

            # Example DAG for the current factors
            example_dag = {
                "smoking": ["lung cancer"],
                "air pollution exposure": ["lung cancer"],
                "exercise habits": ["lung cancer"]
            }
            
            st.markdown("""
            ### 📊 DAG Input Guide
            Enter your DAG structure using either format:
            """)

            # Create two columns for the format examples
            format_col1, format_col2 = st.columns(2)
            
            with format_col1:
                st.markdown("""
                **1. JSON Format:**
                ```json
                {
                    "source": ["target1", "target2"],
                    "source2": ["target3"]
                }
                ```
                """)
            
            with format_col2:
                st.markdown("""
                **2. Python Dict Format:**
                ```python
                {
                    'source': ['target1', 'target2'],
                    'source2': ['target3']
                }
                ```
                """)

            st.markdown("### 💡 Example for Current Factors:")
            st.code(json.dumps(example_dag, indent=2), language='json')

            dag_str = st.text_area(
                "Enter the DAG structure:",
                value=json.dumps(example_dag, indent=2),
                help="Enter the DAG structure as a JSON or Python dictionary",
                height=200
            )

            is_valid, suggested_dag, error_msg = validate_dag_input(dag_str)
            
            if error_msg.startswith("❌"):
                st.error(error_msg)
            elif error_msg.startswith("✅"):
                st.success(error_msg)
                st.markdown("### Current DAG Structure:")
                
                # Display relationships in a more readable format
                relationships = []
                for source, targets in suggested_dag.items():
                    for target in targets:
                        relationships.append(f"• {source} ➜ {target}")
                
                for rel in relationships:
                    st.markdown(rel)

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔍 Critique DAG"):
                        if all_factors and suggested_dag and st.session_state.domain_expertises is not None:
                            suggested_critiques_dag = validator.critique_graph(
                                all_factors, suggested_dag, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                            )
                            st.subheader("📋 DAG Structure Critique:")
                            formatted_critiques = format_critiques(convert_tuples_to_lists(suggested_critiques_dag))
                            st.markdown(formatted_critiques)
                        else:
                            st.warning("Please ensure factors, DAG, and domain expertises are provided.")

                with col2:
                    if st.button("🔍 Find Latent Confounders"):
                        if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                            suggested_latent_confounders = validator.suggest_latent_confounders(
                                treatment, outcome, all_factors, st.session_state.domain_expertises
                            )
                            st.subheader("🎯 Suggested Latent Confounders:")
                            formatted_confounders = format_variables(convert_tuples_to_lists(suggested_latent_confounders))
                            st.markdown(formatted_confounders)
                            
                            # Add explanation for latent confounders
                            st.info("💡 Latent confounders are unmeasured variables that might affect both treatment and outcome. Consider if you can measure these variables or account for them in your analysis.")
                        else:
                            st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

                with col3:
                    if st.button("🔍 Find Negative Controls"):
                        if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                            suggested_negative_controls = validator.suggest_negative_controls(
                                treatment, outcome, all_factors, st.session_state.domain_expertises
                            )
                            st.subheader("🎯 Suggested Negative Controls:")
                            formatted_controls = format_variables(convert_tuples_to_lists(suggested_negative_controls))
                            st.markdown(formatted_controls)
                            
                            # Add explanation for negative controls
                            st.info("💡 Negative controls help validate your causal assumptions. They are variables that should not be affected by your treatment or should not affect your outcome.")
                        else:
                            st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")