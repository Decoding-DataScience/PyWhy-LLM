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
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main Title */
    .main-title {
        color: #1E88E5;
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin: 30px 0 10px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Attribution */
    .attribution {
        text-align: center;
        font-size: 16px;
        color: #424242;
        margin-bottom: 30px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    .attribution a {
        color: #1E88E5;
        text-decoration: none;
    }

    /* Section Headers */
    .section-header {
        color: #2E7D32;
        font-size: 26px;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2E7D32;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Sub-section Headers */
    .section-subtitle {
        color: #1E88E5;
        font-size: 20px;
        font-weight: 500;
        margin: 20px 0 10px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Info Box */
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .info-box ul, .info-box ol {
        margin: 10px 0;
        padding-left: 25px;
    }

    .info-box li {
        padding: 8px 0;
        position: relative;
    }

    /* Remove bullet points styling */
    .info-box ul {
        list-style-type: decimal;
    }

    .info-box li:before {
        content: none;
    }

    /* Info Box Headers */
    .info-box h3 {
        color: #1E88E5;
        font-size: 22px;
        font-weight: 500;
        margin: 0 0 15px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Step Box */
    .step-box {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.6;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .step-box ol {
        margin: 0;
        padding-left: 25px;
    }

    .step-box li {
        padding: 8px 0;
    }

    /* Labels and Text */
    .stTextInput label, .stSelectbox label, .stTextArea label {
        font-size: 16px;
        font-weight: 500;
        color: #2C3E50;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Help Text */
    .stTextInput div[data-baseweb="help"], 
    .stSelectbox div[data-baseweb="help"], 
    .stTextArea div[data-baseweb="help"] {
        font-size: 14px;
        color: #666;
        margin-top: 4px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Buttons */
    .stButton > button {
        font-size: 16px;
        font-weight: 500;
        padding: 10px 20px;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .stButton > button:hover {
        background-color: #1976D2;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: 500;
        color: #2C3E50;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Model Step Title */
    .model-step-title {
        font-size: 30px;
        color: #1E88E5;
        font-weight: 600;
        padding: 15px;
        background-color: #E3F2FD;
        border-radius: 8px;
        margin: 20px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Output Section Title */
    .section-title {
        color: #1E88E5;
        font-size: 24px;
        font-weight: 600;
        margin: 20px 0 15px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Explanation and Recommendation Headers */
    .explanation-box h4, .recommendation-box h4 {
        color: #2c3e50;
        margin: 0 0 12px 0;
        font-size: 18px;
        font-weight: 500;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Output Item Headers */
    .output-item h4 {
        color: #2c3e50;
        margin: 0 0 12px 0;
        font-size: 18px;
        font-weight: 500;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

def apply_custom_css():
    st.markdown("""
        <style>
        /* General Styles */
        .output-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .output-item {
            background: white;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }
        
        .output-item p {
            margin: 5px 0;
            color: #34495e;
        }
        
        /* Confidence Level Styles */
        .confidence-high {
            color: #27ae60;
            font-weight: bold;
        }
        
        .confidence-medium {
            color: #f39c12;
            font-weight: bold;
        }
        
        .confidence-low {
            color: #e74c3c;
            font-weight: bold;
        }
        
        /* Relationship Arrow */
        .relationship-arrow {
            color: #3498db;
            font-weight: normal;
            margin: 0 10px;
        }
        
        /* Explanation and Recommendation Boxes */
        .explanation-box, .recommendation-box {
            background: white;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .explanation-box {
            border-left: 4px solid #3498db;
        }
        
        .recommendation-box {
            border-left: 4px solid #2ecc71;
        }
        
        .explanation-box ul, .recommendation-box ol {
            margin: 10px 0;
            padding-left: 25px;
        }
        
        .explanation-box li, .recommendation-box li {
            margin: 5px 0;
            color: #34495e;
        }
        
        /* Button Styles */
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            background-color: #2c3e50;
            color: white;
            font-weight: 500;
            border: none;
            padding: 10px 15px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #34495e;
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Input Field Styles */
        .stTextInput > div > div > input {
            border-radius: 5px;
            border: 1px solid #bdc3c7;
            padding: 8px 12px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
        }
        
        /* Title and Header Styles */
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            font-weight: 600;
            margin: 25px 0 15px 0;
        }
        
        h3 {
            color: #2c3e50;
            font-size: 1.4em;
            font-weight: 500;
            margin: 20px 0 10px 0;
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

def format_confounder_output(confounders):
    """Format confounders into human-readable text with explanations."""
    if not confounders:
        return "_No confounding variables identified._"
    
    def format_variable_name(name):
        """Format variable names consistently."""
        return str(name).replace('_', ' ').strip().title()
    
    def format_relationship(source, target, score=None):
        """Format a single relationship in human-readable text."""
        source = format_variable_name(source)
        target = format_variable_name(target)
        
        output = f'''
            <div class="output-item">
                <h4>{source} → {target}</h4>
        '''
        
        if score:
            confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            confidence_class = f"confidence-{confidence.lower()}"
            output += f'''
                <div class="{confidence_class}">
                    Confidence Level: {confidence}
                    <br>
                    Relationship Strength: {score:.2f}
                </div>
                <div class="explanation-box">
                    <p>💡 {source} shows a {confidence.lower()} likelihood of being a confounder.</p>
                </div>
            '''
        else:
            output += f'''
                <div class="explanation-box">
                    <p>💡 {source} may act as a confounder through its relationship with {target}.</p>
                </div>
            '''
        
        output += '</div>'
        return output
    
    output = '<div class="output-section">'
    output += '<h3 class="section-title">Identified Confounding Variables</h3>'
    
    if isinstance(confounders, (list, tuple)):
        for item in confounders:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    output += format_relationship(item[0], item[1], item[2] if len(item) > 2 else None)
            else:
                var_name = format_variable_name(item)
                output += f'''
                    <div class="output-item">
                        <h4>{var_name}</h4>
                        <div class="explanation-box">
                            <p>💡 This variable may confound the relationship between treatment and outcome.</p>
                        </div>
                    </div>
                '''
    
    # Add explanation section with improved formatting
    output += '''
        <div class="explanation-section">
            <h4 class="section-subtitle">🔍 Understanding Confounding Variables</h4>
            <div class="explanation-list">
                <ol>
                    <li>Confounders can create spurious associations between variables</li>
                    <li>They affect both the treatment and outcome variables</li>
                    <li>Controlling for confounders is crucial for accurate causal inference</li>
                </ol>
            </div>
        </div>
        
        <div class="recommendation-section">
            <h4 class="section-subtitle">📊 Recommendations</h4>
            <div class="recommendation-list">
                <ol>
                    <li>Include these variables in your data collection plan</li>
                    <li>Use appropriate statistical methods to control for their effects</li>
                    <li>Consider stratification or matching based on these variables</li>
                    <li>Document any unmeasured confounders that might affect your analysis</li>
                </ol>
            </div>
        </div>
    '''
    
    output += '</div>'
    return output

def format_relationship_output(relationships):
    """Format relationships into readable text with explanations."""
    if not relationships:
        return "_No relationships identified._"
    
    output = '<div class="output-section">'
    output += '<h3 class="section-title">Identified Causal Relationships</h3>'
    
    if isinstance(relationships, (list, tuple)):
        for rel in relationships:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                source = str(rel[0]).replace('_', ' ').title()
                target = str(rel[1]).replace('_', ' ').title()
                confidence = 'medium'
                confidence_score = None
                
                if len(rel) >= 3 and isinstance(rel[2], (int, float)):
                    confidence_score = rel[2]
                    confidence = 'high' if confidence_score > 0.7 else 'medium' if confidence_score > 0.4 else 'low'
                
                output += f'''
                    <div class="output-item">
                        <h4>{source} → {target}</h4>
                        <div class="confidence-{confidence}">
                            Confidence Level: {confidence.title()}
                            {f"<br>Relationship Strength: {confidence_score:.2f}" if confidence_score is not None else ""}
                        </div>
                        <div class="recommendation-box">
                            <p><strong>Recommendation:</strong> {get_relationship_recommendation(confidence)}</p>
                        </div>
                    </div>
                '''
    
    # Add explanation section with improved formatting
    output += '''
        <div class="explanation-section">
            <h4 class="section-subtitle">🔍 Understanding These Relationships</h4>
            <ol>
                <li>Strong relationships suggest direct causal effects</li>
                <li>Medium relationships may indicate indirect effects</li>
                <li>Low confidence relationships need further investigation</li>
            </ol>
        </div>
        
        <div class="recommendation-section">
            <h4 class="section-subtitle">📊 Next Steps</h4>
            <ol>
                <li>Focus on strong relationships for primary analysis</li>
                <li>Consider indirect effects in your model</li>
                <li>Validate relationships with domain experts</li>
                <li>Look for potential mediating variables</li>
            </ol>
        </div>
    '''
    
    output += '</div>'
    return output

def get_relationship_recommendation(confidence):
    """Generate recommendations based on confidence level"""
    if confidence == 'high':
        return "Consider this relationship as a primary focus for your causal analysis."
    elif confidence == 'medium':
        return "Further investigation may be needed to strengthen evidence for this relationship."
    else:
        return "Additional data or expert validation recommended before including in analysis."

def get_confidence_level(score):
    """Convert numerical score to confidence level text"""
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    return "Low"

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
    
    output = '<div class="output-section">'
    output += '<h3 class="section-title">Analysis Critiques</h3>'
    
    if isinstance(critiques, dict):
        for category, items in critiques.items():
            output += f'<h4 class="section-subtitle">{str(category).replace("_", " ").title()}</h4>'
            
            if isinstance(items, list):
                output += '<div class="critique-list">'
                for item in items:
                    output += f'''
                        <div class="output-item">
                            <h4>{format_single_critique(item)}</h4>
                            <div class="explanation-box">
                                <p>💡 {generate_critique_explanation(category, item)}</p>
                            </div>
                        </div>
                    '''
                output += '</div>'
            else:
                output += f'''
                    <div class="output-item">
                        <h4>{format_single_critique(items)}</h4>
                        <div class="explanation-box">
                            <p>💡 {generate_critique_explanation(category, items)}</p>
                        </div>
                    </div>
                '''
    elif isinstance(critiques, list):
        output += '<div class="critique-list">'
        for critique in critiques:
            output += f'''
                <div class="output-item">
                    <h4>{format_single_critique(critique)}</h4>
                    <div class="explanation-box">
                        <p>💡 {generate_critique_explanation("general", critique)}</p>
                    </div>
                </div>
            '''
        output += '</div>'
    
    # Add explanation section with improved formatting
    output += '''
        <div class="explanation-section">
            <h4 class="section-subtitle">🔍 Understanding These Critiques</h4>
            <ol>
                <li>Each critique points to potential improvements in your causal model</li>
                <li>Address high-priority critiques first to strengthen your analysis</li>
                <li>Consider the practical feasibility of implementing suggested changes</li>
            </ol>
        </div>
        
        <div class="recommendation-section">
            <h4 class="section-subtitle">📊 Next Steps</h4>
            <ol>
                <li>Review each critique and assess its impact on your analysis</li>
                <li>Prioritize changes based on feasibility and importance</li>
                <li>Document any limitations that cannot be addressed</li>
                <li>Update your model iteratively as you address each point</li>
            </ol>
        </div>
    '''
    
    output += '</div>'
    return output

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
    """Format the confounding variables output with HTML/CSS styling"""
    if not variables:
        return "_No variables identified._"
        
    output = '<div class="output-section">'
    output += '<h3 class="section-title">Identified Variables</h3>'
    
    for var in variables:
        confidence = var.get('confidence', 'medium').lower()
        name = var.get('name', '')
        impact = var.get('impact', '')
        recommendation = var.get('recommendation', '')
        
        output += f'''
            <div class="output-item">
                <h4>{name}</h4>
                <div class="confidence-{confidence}">
                    Confidence Level: {confidence.title()}
                </div>
                <div class="explanation-box">
                    <p><strong>Impact:</strong> {impact}</p>
                </div>
                <div class="recommendation-box">
                    <p><strong>Recommendation:</strong> {recommendation}</p>
                </div>
            </div>
        '''
    
    # Add explanation section
    output += '''
        <div class="explanation-section">
            <h4 class="section-subtitle">🔍 Understanding Variable Impacts</h4>
            <ol>
                <li>High confidence variables should be prioritized in your analysis</li>
                <li>Consider both direct and indirect effects of each variable</li>
                <li>Pay attention to variables with strong theoretical support</li>
            </ol>
        </div>
    '''
    
    output += '</div>'
    return output

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
    # Main title and attribution
    st.markdown('<h1 class="main-title">PyWhy-LLM Causal Analysis Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="attribution">(Created by <a href="https://www.linkedin.com/in/syedalihasannaqvi/" target="_blank">Syed Hasan</a>)</p>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown('<h2 class="section-header">Welcome to PyWhy-LLM</h2>', unsafe_allow_html=True)
    with st.expander("ℹ️ What is PyWhy-LLM?", expanded=True):
        st.markdown("""
        <div class="info-box">
        <p>PyWhy-LLM is an innovative tool that combines Large Language Models (LLMs) with causal analysis to help researchers and analysts:</p>
        <ul>
            <li>Identify potential causal relationships</li>
            <li>Suggest confounding variables</li>
            <li>Validate causal assumptions</li>
            <li>Build and critique DAGs (Directed Acyclic Graphs)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown('<h2 class="section-header">Quick Start Guide</h2>', unsafe_allow_html=True)
    with st.expander("📚 How to Use PyWhy-LLM", expanded=True):
        st.markdown("""
        <div class="step-box">
        <ol>
            <li>Select your analysis step from the sidebar</li>
            <li>Enter your variables and factors in the input fields</li>
            <li>Follow the step-by-step process for your chosen analysis</li>
            <li>Review and interpret the results</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Main Analysis Interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="section-header">Analysis Configuration</h2>', unsafe_allow_html=True)
        
        llm_model = st.selectbox("Choose LLM Model", ["gpt-4"])
        
        analysis_type = st.selectbox(
            "📊 Choose Analysis Step",
            ["Model Suggestion", "Identification Suggestion", "Validation Suggestion"]
        )
        
        st.markdown('<h3 class="section-header">Variables Input</h3>', unsafe_allow_html=True)
        
        factors_help = """
        Examples by domain:
        • Medical: "smoking, lung cancer, exercise habits, air pollution"
        • Education: "study hours, test scores, sleep quality, stress"
        • Economics: "interest rates, inflation, unemployment, gdp"
        • Environmental: "co2 emissions, temperature, deforestation, rainfall"
        """
        
        default_factors = "smoking, lung cancer, exercise habits, air pollution exposure"
        all_factors_str = st.text_area(
            "📝 Enter all relevant factors (comma-separated):", 
            value=default_factors,
            help=factors_help
        )
        all_factors = [factor.strip() for factor in all_factors_str.split(',')]

        treatment = st.text_input(
            "🎯 Enter the treatment variable:",
            value="smoking",
            help="The variable whose effect you want to study"
        )

        outcome = st.text_input(
            "🎯 Enter the outcome variable:",
            value="lung cancer",
            help="The variable you want to measure the effect on"
        )

    with col2:
        st.markdown('<h2 class="section-header">Analysis Steps</h2>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'domain_expertises' not in st.session_state:
            st.session_state.domain_expertises = None

        if analysis_type == "Model Suggestion":
            st.markdown('<div class="model-step-title">🔍 Model Suggestion Step</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Build your initial causal model through these steps:
            <ol>
                <li>Identify required domain expertise</li>
                <li>Discover potential confounding variables</li>
                <li>Establish pair-wise relationships between variables</li>
            </ol>
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
                    st.subheader("Potential Confounding Variables")
                    formatted_confounders = format_confounder_output(convert_tuples_to_lists(suggested_confounders))
                    st.markdown(formatted_confounders)
                else:
                    st.warning("Please ensure all required information is provided.")

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
            <ol>
                <li>Find backdoor adjustment sets</li>
                <li>Discover potential mediator variables</li>
                <li>Locate possible instrumental variables</li>
            </ol>
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
            <ol>
                <li>Validate your DAG structure</li>
                <li>Identify potential latent confounders</li>
                <li>Find suitable negative controls</li>
            </ol>
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

                # Create a row for all buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    critique_clicked = st.button("🔍 Critique DAG")
                with col2:
                    confounders_clicked = st.button("🔍 Find Latent Confounders")
                with col3:
                    controls_clicked = st.button("🔍 Find Negative Controls")
                
                # Create a container for results below all buttons
                results_container = st.container()
                
                with results_container:
                    if critique_clicked:
                        if all_factors and suggested_dag and st.session_state.domain_expertises is not None:
                            suggested_critiques_dag = validator.critique_graph(
                                all_factors, suggested_dag, st.session_state.domain_expertises, RelationshipStrategy.Pairwise
                            )
                            st.markdown("### 📋 DAG Structure Critique")
                            formatted_critiques = format_critiques(convert_tuples_to_lists(suggested_critiques_dag))
                            st.markdown(formatted_critiques)
                        else:
                            st.warning("Please ensure factors, DAG, and domain expertises are provided.")

                    if confounders_clicked:
                        if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                            suggested_latent_confounders = validator.suggest_latent_confounders(
                                treatment, outcome, all_factors, st.session_state.domain_expertises
                            )
                            st.markdown("### 🎯 Potential Latent Confounders")
                            formatted_confounders = format_variables(convert_tuples_to_lists(suggested_latent_confounders))
                            st.markdown(formatted_confounders)
                            
                            # Add explanation for latent confounders
                            st.info("💡 Latent confounders are unmeasured variables that might affect both treatment and outcome. Consider if you can measure these variables or account for them in your analysis.")
                        else:
                            st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")

                    if controls_clicked:
                        if treatment and outcome and all_factors and st.session_state.domain_expertises is not None:
                            suggested_negative_controls = validator.suggest_negative_controls(
                                treatment, outcome, all_factors, st.session_state.domain_expertises
                            )
                            st.markdown("### 🎯 Suggested Negative Controls")
                            formatted_controls = format_variables(convert_tuples_to_lists(suggested_negative_controls))
                            st.markdown(formatted_controls)
                            
                            # Add explanation for negative controls
                            st.info("💡 Negative controls help validate your causal assumptions. They are variables that should not be affected by your treatment or should not affect your outcome.")
                        else:
                            st.warning("Please ensure treatment, outcome, factors, and domain expertises are provided.")