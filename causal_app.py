import streamlit as st
from pywhyllm.suggesters.model_suggester import ModelSuggester
from pywhyllm.suggesters.identification_suggester import IdentificationSuggester
from pywhyllm.suggesters.validation_suggester import ValidationSuggester
from pywhyllm import RelationshipStrategy
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import graphviz

# Standard error messages
OPENAI_API_KEY_ERROR = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
MISSING_VARIABLES_ERROR = "Please enter all required variables (factors, treatment, and outcome)."
MISSING_FACTORS_ERROR = "Please enter some factors first."

# Initialize OpenAI client
def get_openai_client():
    """Get OpenAI client with proper error handling."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(OPENAI_API_KEY_ERROR)
        return None
    return OpenAI(api_key=api_key)

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
    """Convert all tuples to lists recursively in a nested structure."""
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
        return st.markdown("_No confounding variables identified._")
    
    def format_variable_name(name):
        """Format variable names consistently."""
        return str(name).replace('_', ' ').strip().title()
    
    def format_relationship(source, target, score=None):
        """Format a single relationship in human-readable text."""
        source = format_variable_name(source)
        target = format_variable_name(target)
        
        st.markdown(f"### {source} → {target}")
        
        if score:
            confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            confidence_color = "#27ae60" if confidence == "High" else "#f39c12" if confidence == "Medium" else "#e74c3c"
            
            st.markdown(f"""
                <div style='color: {confidence_color}; font-weight: bold; margin-bottom: 10px;'>
                    Confidence Level: {confidence}
                    <br>
                    Relationship Strength: {score:.2f}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"💡 {source} shows a {confidence.lower()} likelihood of being a confounder.")
        else:
            st.markdown(f"💡 {source} may act as a confounder through its relationship with {target}.")
    
    st.markdown("## Identified Confounding Variables")
    
    if isinstance(confounders, (list, tuple)):
        for item in confounders:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    format_relationship(item[0], item[1], item[2] if len(item) > 2 else None)
            else:
                var_name = format_variable_name(item)
                st.markdown(f"### {var_name}")
                st.markdown(f"💡 This variable may confound the relationship between treatment and outcome.")
    
    # Add explanation section
    st.markdown("### 🔍 Understanding Confounding Variables")
    st.markdown("""
    1. Confounders can create spurious associations between variables
    2. They affect both the treatment and outcome variables
    3. Controlling for confounders is crucial for accurate causal inference
    """)
    
    # Add recommendations
    st.markdown("### 📊 Recommendations")
    st.markdown("""
    1. Include these variables in your data collection plan
    2. Use appropriate statistical methods to control for their effects
    3. Consider stratification or matching based on these variables
    4. Document any unmeasured confounders that might affect your analysis
    """)

def format_list_to_text(items):
    """Convert a list of items into a readable text format."""
    if not items:
        return ""
    
    # Clean up each item and remove quotes and brackets
    cleaned_items = [str(item).strip("'[]{}").strip() for item in items]
    
    if len(cleaned_items) == 1:
        return cleaned_items[0]
    elif len(cleaned_items) == 2:
        return f"{cleaned_items[0]} and {cleaned_items[1]}"
    else:
        return ", ".join(cleaned_items[:-1]) + f", and {cleaned_items[-1]}"

def format_variables(variables):
    """Format the variables output with proper styling"""
    if not variables:
        return st.markdown("_No variables identified._")
    
    try:
        # Convert tuples to lists if needed
        variables = convert_tuples_to_lists(variables)
        
        st.markdown("## Identified Variables")
        
        if isinstance(variables, (dict, list)):
            # Handle dictionary format
            if isinstance(variables, dict):
                variables = [{"name": k, "value": v} for k, v in variables.items()]
            
            for var in variables:
                if isinstance(var, dict):
                    # Handle dictionary format
                    name = var.get('name', '')
                    confidence = var.get('confidence', 'medium').lower()
                    impact = var.get('impact', [])
                    recommendation = var.get('recommendation', [])
                    
                    # Convert impact and recommendation to readable text
                    if isinstance(impact, (list, tuple)):
                        impact = format_list_to_text(impact)
                    if isinstance(recommendation, (list, tuple)):
                        recommendation = format_list_to_text(recommendation)
                    
                elif isinstance(var, (list, tuple)):
                    # Handle list/tuple format
                    name = str(var[0]) if len(var) > 0 else ''
                    confidence = 'medium'
                    impact = format_list_to_text(var[1]) if len(var) > 1 else ''
                    recommendation = format_list_to_text(var[2]) if len(var) > 2 else ''
                else:
                    # Handle single string/value
                    name = str(var).strip("'[]{}").strip()
                    confidence = 'medium'
                    impact = ''
                    recommendation = ''
                
                confidence_color = "#27ae60" if confidence == "high" else "#f39c12" if confidence == "medium" else "#e74c3c"
                
                st.markdown(f"### {name}")
                st.markdown(f"""
                    <div style='color: {confidence_color}; font-weight: bold; margin-bottom: 10px;'>
                        Confidence Level: {confidence.title()}
                    </div>
                """, unsafe_allow_html=True)
                
                if impact:
                    st.markdown(f"**Key Impacts:** {impact}")
                if recommendation:
                    st.markdown(f"**Recommendations:** Consider {recommendation} in your analysis")
        
        # Add explanation section
        st.markdown("### 🔍 Understanding Variable Impacts")
        st.markdown("""
        1. High confidence variables should be prioritized in your analysis
        2. Consider both direct and indirect effects of each variable
        3. Pay attention to variables with strong theoretical support
        """)
        
    except Exception as e:
        st.error("""
            Unable to format variables. Please ensure the input is in the correct format.
            The variables should be provided as a list of factors with their impacts and recommendations.
        """)
        return None

def format_backdoor_set(backdoor_set):
    """Format the backdoor set with proper styling"""
    if not backdoor_set:
        return st.markdown("_No backdoor adjustment set identified._")
    
    try:
        # Convert tuples to lists if needed
        backdoor_set = convert_tuples_to_lists(backdoor_set)
        
        st.markdown("## Suggested Backdoor Set")
        
        # Create a visual representation of the backdoor set
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Define node styles
        dot.attr('node', 
            shape='rect',
            style='rounded,filled',
            fillcolor='white',
            fontname='Arial',
            margin='0.3,0.2'
        )
        
        # Define edge styles
        dot.attr('edge',
            color='#1E88E5',
            penwidth='2'
        )
        
        # Track nodes to avoid duplicates
        nodes = set()
        
        # Add nodes and edges for the backdoor set
        if isinstance(backdoor_set, (list, tuple)):
            for var in backdoor_set:
                if isinstance(var, dict):
                    name = var.get('name', '')
                    confidence = var.get('confidence', 'medium')
                    if name:
                        if name not in nodes:
                            dot.node(name, name)
                            nodes.add(name)
                            
                        # Format the variable details
                        st.markdown(f"### {name}")
                        confidence_color = "#27ae60" if confidence == "high" else "#f39c12" if confidence == "medium" else "#e74c3c"
                        st.markdown(f"""
                            <div style='color: {confidence_color}; font-weight: bold; margin-bottom: 10px;'>
                                Confidence Level: {confidence.title()}
                            </div>
                        """, unsafe_allow_html=True)
                        
                elif isinstance(var, (list, tuple)):
                    # Handle relationship format
                    if len(var) >= 2:
                        source = str(var[0])
                        target = str(var[1])
                        
                        # Add nodes if they don't exist
                        if source not in nodes:
                            dot.node(source, source)
                            nodes.add(source)
                        if target not in nodes:
                            dot.node(target, target)
                            nodes.add(target)
                            
                        # Add edge
                        dot.edge(source, target)
                        
                        # Format the relationship
                        st.markdown(f"### {source} → {target}")
                        st.markdown("This relationship is part of the backdoor adjustment set.")
                else:
                    # Handle single variable
                    var_name = str(var)
                    if var_name not in nodes:
                        dot.node(var_name, var_name)
                        nodes.add(var_name)
                    st.markdown(f"### {var_name}")
                    st.markdown("This variable should be included in the backdoor adjustment set.")
        
        # Display the visualization
        if nodes:
            st.graphviz_chart(dot)
        
        # Add explanation section
        with st.expander("🔍 Understanding Backdoor Adjustment", expanded=True):
            st.markdown("""
            ### What is Backdoor Adjustment?
            Backdoor adjustment helps control for confounding variables in causal analysis by:
            1. Identifying variables that affect both treatment and outcome
            2. Blocking "backdoor paths" that create spurious associations
            3. Enabling unbiased estimation of causal effects
            
            ### How to Use These Variables
            - Include these variables in your analysis model
            - Collect data on these variables during your study
            - Consider stratification or matching based on these variables
            - Document any unmeasured confounders
            """)
        
        # Add recommendations
        with st.expander("📊 Recommendations", expanded=True):
            st.markdown("""
            1. Prioritize collecting data on high-confidence backdoor variables
            2. Use appropriate statistical methods to control for these variables
            3. Consider both direct and indirect paths in your analysis
            4. Validate the completeness of the backdoor set with domain experts
            """)
        
    except Exception as e:
        st.error(f"""
            Unable to format backdoor set. Please ensure the input is in the correct format.
            Expected format: List of variables or relationships that form the backdoor adjustment set.
            Error: {str(e)}
        """)
        return None

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

def create_dag_visualization(relationships):
    """Create a visual DAG using Graphviz."""
    if not relationships:
        return None
    
    # Create a new directed graph
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Define node styles
    dot.attr('node', 
        shape='rect',
        style='rounded,filled',
        fillcolor='white',
        fontname='Arial',
        margin='0.3,0.2'
    )
    
    # Define edge styles
    dot.attr('edge',
        color='#1E88E5',
        penwidth='2'
    )
    
    # Track nodes to avoid duplicates
    nodes = set()
    
    # Add nodes and edges
    for rel in relationships:
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            source = str(rel[0]).strip()
            target = str(rel[1]).strip()
            
            # Add nodes if they don't exist
            if source not in nodes:
                dot.node(source, source)
                nodes.add(source)
            if target not in nodes:
                dot.node(target, target)
                nodes.add(target)
            
            # Add edge with confidence as label if available
            if len(rel) > 2:
                confidence = float(rel[2])
                # Only show confidence label if it's significant
                if confidence > 0.5:
                    dot.edge(source, target, label=f" {confidence:.2f}")
                else:
                    dot.edge(source, target)
            else:
                dot.edge(source, target)
    
    return dot

def format_relationship_output(relationships):
    """Format relationships into readable text with explanations and visualization."""
    if not relationships:
        return st.markdown("_No relationships identified._")
    
    try:
        # Create and display the DAG visualization
        dot = create_dag_visualization(relationships)
        if dot:
            st.graphviz_chart(dot)
        
        st.markdown("## Detailed Relationship Analysis")
        
        for rel in relationships:
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                source = str(rel[0]).strip()
                target = str(rel[1]).strip()
                confidence_score = float(rel[2]) if len(rel) > 2 else 0.5
                
                # Determine confidence level and color
                if confidence_score > 0.7:
                    confidence = "high"
                    confidence_color = "#27ae60"
                elif confidence_score > 0.4:
                    confidence = "medium"
                    confidence_color = "#f39c12"
                else:
                    confidence = "low"
                    confidence_color = "#e74c3c"
                
                # Create expandable section for each relationship
                with st.expander(f"🔍 {source} → {target}"):
                    st.markdown(f"""
                        <div style='color: {confidence_color}; font-weight: bold; margin-bottom: 10px;'>
                            Confidence Level: {confidence.title()}
                            <br>
                            Relationship Strength: {confidence_score:.2f}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add relationship explanation
                    explanation = get_relationship_explanation(source, target, confidence)
                    st.markdown(f"**Analysis:** {explanation}")
                    
                    # Add recommendation
                    recommendation = get_relationship_recommendation(confidence)
                    st.markdown(f"**Recommendation:** {recommendation}")
        
        # Add explanation section
        with st.expander("🔍 Understanding These Relationships", expanded=False):
            st.markdown("""
            ### Interpreting the Diagram
            1. **Nodes:** Represent variables in your causal model
            2. **Arrows:** Show the direction of causal influence
            3. **Numbers:** Indicate the strength of relationships (0-1)
            
            ### Confidence Levels
            - **High (>0.7):** Strong evidence of direct causal effect
            - **Medium (0.4-0.7):** Moderate evidence, possible indirect effects
            - **Low (<0.4):** Weak evidence, needs further investigation
            """)
        
        # Add next steps
        with st.expander("📊 Next Steps", expanded=False):
            st.markdown("""
            1. Focus on high-confidence relationships for primary analysis
            2. Consider indirect effects through medium-confidence paths
            3. Validate relationships with domain experts
            4. Look for potential mediating variables
            """)
        
    except Exception as e:
        st.error(f"Error formatting relationships: {str(e)}")
        return None

def get_relationship_explanation(source, target, confidence):
    """Generate explanations for relationships based on confidence level"""
    if confidence == 'high':
        return f"Strong evidence suggests that changes in {source} directly affect {target}."
    elif confidence == 'medium':
        return f"There appears to be a moderate relationship between {source} and {target}, possibly involving other factors."
    else:
        return f"The relationship between {source} and {target} requires further investigation to establish causality."

def get_relationship_recommendation(confidence):
    """Generate recommendations based on confidence level"""
    if confidence == 'high':
        return "Consider this relationship as a primary focus for your causal analysis."
    elif confidence == 'medium':
        return "Further investigation may be needed to strengthen evidence for this relationship."
    else:
        return "Additional data or expert validation recommended before including in analysis."

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
        return st.markdown("_No critiques found._")
    
    def format_single_critique(critique):
        """Format a single critique into readable text."""
        return str(critique).strip("'[]{}").strip().replace('_', ' ').title()
    
    st.markdown("## Analysis Critiques")
    
    if isinstance(critiques, dict):
        for category, items in critiques.items():
            category = str(category).strip("'[]{}").strip().replace('_', ' ').title()
            st.markdown(f"### {category}")
            
            if isinstance(items, list):
                for item in items:
                    critique_text = format_single_critique(item)
                    st.markdown(f"#### {critique_text}")
                    st.markdown(f"💡 {generate_critique_explanation(category, critique_text)}")
            else:
                critique_text = format_single_critique(items)
                st.markdown(f"#### {critique_text}")
                st.markdown(f"💡 {generate_critique_explanation(category, critique_text)}")
    elif isinstance(critiques, list):
        for critique in critiques:
            critique_text = format_single_critique(critique)
            st.markdown(f"#### {critique_text}")
            st.markdown(f"💡 {generate_critique_explanation('general', critique_text)}")
    
    # Add explanation section
    st.markdown("### 🔍 Understanding These Critiques")
    st.markdown("""
    1. Each critique points to potential improvements in your causal model
    2. Address high-priority critiques first to strengthen your analysis
    3. Consider the practical feasibility of implementing suggested changes
    """)
    
    # Add next steps
    st.markdown("### 📊 Next Steps")
    st.markdown("""
    1. Review each critique and assess its impact on your analysis
    2. Prioritize changes based on feasibility and importance
    3. Document any limitations that cannot be addressed
    4. Update your model iteratively as you address each point
    """)

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

def suggest_variables_from_factors(factors, openai_api_key):
    """Use OpenAI to suggest treatment and outcome variables from the input factors."""
    if not factors:
        return None, None
    
    try:
        client = get_openai_client()
        if not client:
            return None, None
            
        # Create a prompt for the OpenAI API
        prompt = f"""Given these factors in a causal analysis context: {', '.join(factors)}

For a causal analysis similar to the example of how parental income and tutoring affect school quality and ultimately job offers through college admission, please identify:
1. The most likely treatment variable (the cause/intervention)
2. The most likely outcome variable (the final effect to measure)

Consider the natural flow of causation and choose variables that would have a meaningful causal relationship.

Format your response exactly like this example:
treatment: School Quality
outcome: Job Offer

Your response:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert helping to identify treatment and outcome variables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        # Parse the response
        suggestion = response.choices[0].message.content.strip()
        lines = suggestion.split('\n')
        
        treatment = None
        outcome = None
        
        for line in lines:
            if line.startswith('treatment:'):
                treatment = line.replace('treatment:', '').strip()
            elif line.startswith('outcome:'):
                outcome = line.replace('outcome:', '').strip()
        
        return treatment, outcome
        
    except Exception as e:
        error_msg = str(e)
        if "openai.ChatCompletion" in error_msg:
            st.error("""
                OpenAI API version mismatch. Please update your openai package:
                1. Run: pip install --upgrade openai
                2. Restart the application
            """)
        else:
            st.error(f"Error suggesting variables: {error_msg}")
        return None, None

def suggest_confounders_from_factors(treatment, outcome, factors, openai_api_key):
    """Use OpenAI to suggest potential confounding variables."""
    if not factors or not treatment or not outcome:
        return None
    
    try:
        client = get_openai_client()
        if not client:
            return None

        prompt = f"""Given:
- Treatment variable: {treatment}
- Outcome variable: {outcome}
- All factors: {', '.join(factors)}

Please identify potential confounding variables that might affect both the treatment and outcome.
Consider variables that could create spurious associations.

Format your response as a list of confounders with confidence levels (high/medium/low) like this:
{{"variable1": "high", "variable2": "medium"}}

Your response:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert helping to identify confounding variables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Parse the response
        try:
            suggestion = response.choices[0].message.content.strip()
            confounders = json.loads(suggestion)
            return confounders
        except json.JSONDecodeError:
            st.error("Error parsing the confounders suggestion. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Error suggesting confounders: {str(e)}")
        return None

def suggest_relationships_from_factors(treatment, outcome, factors, openai_api_key):
    """Use OpenAI to suggest pair-wise relationships for DAG."""
    if not factors or not treatment or not outcome:
        return None
    
    try:
        client = get_openai_client()
        if not client:
            return None

        # Create a more structured prompt
        prompt = f"""Given these variables in a causal analysis context:
- Treatment: {treatment}
- Outcome: {outcome}
- Other factors: {', '.join(f for f in factors if f not in [treatment, outcome])}

Please identify potential causal relationships between these variables.
Focus on direct relationships and provide confidence scores.

Format your response EXACTLY as a list of lists, where each inner list contains:
1. Source variable (string)
2. Target variable (string)
3. Confidence score (number between 0 and 1)

Example format:
[
    ["{treatment}", "{outcome}", 0.8],
    ["factor1", "{outcome}", 0.6],
    ["{treatment}", "factor2", 0.7]
]

Ensure:
1. Each relationship is a direct causal link
2. Confidence scores reflect the strength of evidence (0-1)
3. Include relationships involving treatment and outcome
4. Only include relationships with reasonable causal basis

Your response:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert. Provide relationships in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse the response
        try:
            suggestion = response.choices[0].message.content.strip()
            
            # Clean up the response to handle common formatting issues
            suggestion = suggestion.replace("'", '"')  # Replace single quotes with double quotes
            suggestion = suggestion.replace("None", "null")  # Replace Python None with JSON null
            
            # Extract the list part from the response if there's additional text
            import re
            list_pattern = r'\[([\s\S]*)\]'  # Match everything between first [ and last ]
            match = re.search(list_pattern, suggestion)
            if match:
                suggestion = f"[{match.group(1)}]"
            
            try:
                # Try parsing as JSON first
                relationships = json.loads(suggestion)
            except json.JSONDecodeError:
                # If JSON parsing fails, try evaluating as Python literal
                import ast
                relationships = ast.literal_eval(suggestion)
            
            # Validate and format relationships
            formatted_relationships = []
            if isinstance(relationships, list):
                for rel in relationships:
                    if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                        # Ensure proper string formatting and numerical confidence
                        source = str(rel[0]).strip()
                        target = str(rel[1]).strip()
                        confidence = float(rel[2]) if len(rel) > 2 and rel[2] is not None else 0.5
                        confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
                        formatted_relationships.append([source, target, confidence])
            
            if formatted_relationships:
                return formatted_relationships
            else:
                st.warning("No valid relationships could be extracted from the model's response.")
                return None
                
        except Exception as e:
            st.error(f"Error parsing relationships: {str(e)}")
            st.info("The model's response could not be properly parsed. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Error suggesting relationships: {str(e)}")
        return None

def suggest_backdoor_from_factors(treatment, outcome, factors, openai_api_key):
    """Use OpenAI to suggest backdoor adjustment set."""
    if not factors or not treatment or not outcome:
        return None
    
    try:
        client = get_openai_client()
        if not client:
            return None

        prompt = f"""Given a causal analysis with:
Treatment: {treatment}
Outcome: {outcome}
All factors: {', '.join(factors)}

Please identify the backdoor adjustment set - variables that should be controlled for to estimate the causal effect of {treatment} on {outcome}.

Consider:
1. Variables that affect both treatment and outcome
2. Variables that create backdoor paths
3. Variables that might confound the relationship

Format your response as a list of variables with their roles, like this:
[
    ["parental_income", "affects both school quality and college admission"],
    ["tutoring", "mediates between income and school quality"],
    ["student_motivation", "affects both school performance and job prospects"]
]

Your response:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert helping to identify backdoor adjustment sets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse the response
        try:
            suggestion = response.choices[0].message.content.strip()
            
            # Clean up the response
            suggestion = suggestion.replace("'", '"')
            
            # Extract the list part from the response
            import re
            list_pattern = r'\[([\s\S]*)\]'
            match = re.search(list_pattern, suggestion)
            if match:
                suggestion = f"[{match.group(1)}]"
            
            # Parse the response
            try:
                backdoor_set = json.loads(suggestion)
            except json.JSONDecodeError:
                import ast
                backdoor_set = ast.literal_eval(suggestion)
            
            # Format the backdoor set
            formatted_set = []
            if isinstance(backdoor_set, list):
                for var in backdoor_set:
                    if isinstance(var, (list, tuple)) and len(var) >= 2:
                        name = str(var[0]).strip()
                        explanation = str(var[1]).strip()
                        formatted_set.append({
                            "name": name,
                            "explanation": explanation,
                            "confidence": "high" if "both" in explanation.lower() else "medium"
                        })
            
            return formatted_set if formatted_set else None
            
        except Exception as e:
            st.error(f"Error parsing backdoor set suggestion: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error suggesting backdoor set: {str(e)}")
        return None

def suggest_mediator_from_factors(treatment, outcome, factors, openai_api_key):
    """Use OpenAI to suggest mediator variables."""
    if not factors or not treatment or not outcome:
        return None
    
    try:
        client = get_openai_client()
        if not client:
            return None

        prompt = f"""Given a causal analysis with:
Treatment: {treatment}
Outcome: {outcome}
All factors: {', '.join(factors)}

Please identify potential mediator variables - variables that lie on the causal path between {treatment} and {outcome}.

Consider the example of how school quality affects job offers through college admission:
- College admission is a mediator because:
  1. School quality affects college admission chances
  2. College admission then affects job offers
  3. It's on the causal path between treatment and outcome

Format your response EXACTLY as a JSON array of arrays, where each inner array contains:
1. The name of the mediator variable (string)
2. A brief explanation of its mediating role (string)
3. A confidence score between 0 and 1 (number)

Example format:
[
    ["college_admission", "mediates between school quality and job offers", 0.8],
    ["academic_performance", "links school quality to college prospects", 0.7]
]

Ensure your response is a valid JSON array and includes ONLY the array, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert. Return ONLY the requested JSON array format, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        try:
            suggestion = response.choices[0].message.content.strip()
            
            # Clean up the response
            suggestion = suggestion.replace("'", '"')  # Replace single quotes with double quotes
            suggestion = suggestion.replace("\n", " ")  # Remove newlines
            
            # Extract just the array part if there's additional text
            import re
            array_pattern = r'\[(.*)\]'
            match = re.search(array_pattern, suggestion)
            if match:
                suggestion = f"[{match.group(1)}]"
            
            try:
                # Try parsing as JSON first
                mediators = json.loads(suggestion)
                
                # Validate the structure
                if not isinstance(mediators, list):
                    st.warning("Invalid response format. Expected a list of mediator variables.")
                    return None
                
                for mediator in mediators:
                    if not isinstance(mediator, list) or len(mediator) < 3:
                        st.warning("Invalid mediator format. Each mediator should have a name, explanation, and score.")
                        return None
                    
                    # Ensure score is a float between 0 and 1
                    mediator[2] = float(mediator[2])
                    if not 0 <= mediator[2] <= 1:
                        mediator[2] = max(0, min(1, mediator[2]))  # Clamp between 0 and 1
                
                return mediators
                
            except json.JSONDecodeError:
                # If JSON fails, try Python literal evaluation
                import ast
                try:
                    mediators = ast.literal_eval(suggestion)
                    
                    # Apply the same validation as above
                    if not isinstance(mediators, list):
                        st.warning("Invalid response format. Expected a list of mediator variables.")
                        return None
                    
                    for mediator in mediators:
                        if not isinstance(mediator, list) or len(mediator) < 3:
                            st.warning("Invalid mediator format. Each mediator should have a name, explanation, and score.")
                            return None
                        
                        # Ensure score is a float between 0 and 1
                        mediator[2] = float(mediator[2])
                        if not 0 <= mediator[2] <= 1:
                            mediator[2] = max(0, min(1, mediator[2]))  # Clamp between 0 and 1
                    
                    return mediators
                except:
                    st.warning("Could not parse the mediator suggestions. Please try again.")
                    return None
            
        except Exception as e:
            st.error(f"Error processing mediator suggestions: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error suggesting mediators: {str(e)}")
        return None

def suggest_iv_from_factors(treatment, outcome, factors, openai_api_key):
    """Use OpenAI to suggest instrumental variables."""
    if not factors or not treatment or not outcome:
        return None
    
    try:
        client = get_openai_client()
        if not client:
            return None

        prompt = f"""Given a causal analysis with:
Treatment: {treatment}
Outcome: {outcome}
All factors: {', '.join(factors)}

Please identify potential instrumental variables (IVs) that could help estimate the causal effect of {treatment} on {outcome}.

A good instrumental variable:
1. Affects the treatment variable
2. Only affects the outcome through the treatment
3. Is not affected by any confounders of the treatment-outcome relationship

For example, in an education study:
- Distance to high-quality schools could be an IV for school quality
- It affects which school a student attends
- It likely only affects job prospects through its effect on school quality
- It's typically not related to other factors affecting job success

Format your response EXACTLY as a JSON array of arrays, where each inner array contains:
1. The name of the instrumental variable (string)
2. A brief explanation of why it's a good IV (string)
3. A validity score between 0 and 1 (number)

Example format:
[
    ["distance_to_schools", "affects school choice but not directly related to job outcomes", 0.85],
    ["local_education_policy", "influences school quality but not directly linked to employment", 0.75]
]

Ensure your response is a valid JSON array and includes ONLY the array, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a causal inference expert. Return ONLY the requested JSON array format, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        try:
            suggestion = response.choices[0].message.content.strip()
            
            # Clean up the response
            suggestion = suggestion.replace("'", '"')  # Replace single quotes with double quotes
            suggestion = suggestion.replace("\n", " ")  # Remove newlines
            
            # Extract just the array part if there's additional text
            import re
            array_pattern = r'\[(.*)\]'
            match = re.search(array_pattern, suggestion)
            if match:
                suggestion = f"[{match.group(1)}]"
            
            try:
                # Try parsing as JSON first
                ivs = json.loads(suggestion)
                
                # Validate the structure
                if not isinstance(ivs, list):
                    st.warning("Invalid response format. Expected a list of instrumental variables.")
                    return None
                
                for iv in ivs:
                    if not isinstance(iv, list) or len(iv) < 3:
                        st.warning("Invalid IV format. Each IV should have a name, explanation, and score.")
                        return None
                    
                    # Ensure score is a float between 0 and 1
                    iv[2] = float(iv[2])
                    if not 0 <= iv[2] <= 1:
                        iv[2] = max(0, min(1, iv[2]))  # Clamp between 0 and 1
                
                return ivs
                
            except json.JSONDecodeError:
                # If JSON fails, try Python literal evaluation
                import ast
                try:
                    ivs = ast.literal_eval(suggestion)
                    
                    # Apply the same validation as above
                    if not isinstance(ivs, list):
                        st.warning("Invalid response format. Expected a list of instrumental variables.")
                        return None
                    
                    for iv in ivs:
                        if not isinstance(iv, list) or len(iv) < 3:
                            st.warning("Invalid IV format. Each IV should have a name, explanation, and score.")
                            return None
                        
                        # Ensure score is a float between 0 and 1
                        iv[2] = float(iv[2])
                        if not 0 <= iv[2] <= 1:
                            iv[2] = max(0, min(1, iv[2]))  # Clamp between 0 and 1
                    
                    return ivs
                except:
                    st.warning("Could not parse the IV suggestions. Please try again.")
                    return None
            
        except Exception as e:
            st.error(f"Error processing IV suggestions: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error suggesting IVs: {str(e)}")
        return None

def format_mediator_output(mediators):
    """Format mediators into readable text with visualization."""
    if not mediators:
        return st.markdown("_No mediator variables identified._")
    
    try:
        # Create visualization
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        
        # Node styles
        dot.attr('node',
            shape='rect',
            style='rounded,filled',
            fillcolor='white',
            fontname='Arial',
            margin='0.3,0.2'
        )
        
        # Edge styles
        dot.attr('edge',
            color='#1E88E5',
            penwidth='2'
        )
        
        # Add treatment and outcome nodes
        treatment = st.session_state.treatment_input
        outcome = st.session_state.outcome_input
        dot.node(treatment, treatment)
        dot.node(outcome, outcome)
        
        # Add mediator nodes and edges
        for med in mediators:
            if isinstance(med, (list, tuple)) and len(med) >= 2:
                mediator = str(med[0]).strip()
                explanation = str(med[1]).strip()
                confidence = float(med[2]) if len(med) > 2 else 0.5
                
                # Add mediator node
                dot.node(mediator, mediator)
                
                # Add edges
                dot.edge(treatment, mediator)
                dot.edge(mediator, outcome)
                
                # Format confidence level
                if confidence > 0.7:
                    confidence_level = "high"
                    confidence_color = "#27ae60"
                elif confidence > 0.4:
                    confidence_level = "medium"
                    confidence_color = "#f39c12"
                else:
                    confidence_level = "low"
                    confidence_color = "#e74c3c"
                
                # Display mediator information
                with st.expander(f"🔄 {mediator} (Mediator)", expanded=True):
                    st.markdown(f"""
                        <div style='color: {confidence_color}; font-weight: bold; margin-bottom: 10px;'>
                            Confidence Level: {confidence_level.title()}
                            <br>
                            Mediation Strength: {confidence:.2f}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Role:** {explanation}")
                    st.markdown("""
                        **Mediation Path:**
                        1. Treatment → Mediator
                        2. Mediator → Outcome
                    """)
        
        # Display the visualization
        st.graphviz_chart(dot)
        
        # Add explanation
        with st.expander("🔍 Understanding Mediation", expanded=True):
            st.markdown("""
            ### What are Mediator Variables?
            Mediator variables help explain *how* or *why* the treatment affects the outcome:
            1. They are affected by the treatment
            2. They in turn affect the outcome
            3. They represent the mechanism of the causal effect
            
            ### How to Use Mediators
            - Include them in path analysis
            - Test for indirect effects
            - Consider them in intervention design
            - Use them to understand causal mechanisms
            """)
        
    except Exception as e:
        st.error(f"Error formatting mediators: {str(e)}")
        return None

def format_iv_output(ivs):
    """Format instrumental variables into readable text with visualization."""
    if not ivs:
        return st.markdown("_No instrumental variables identified._")
    
    try:
        # Create visualization
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        
        # Node styles
        dot.attr('node',
            shape='rect',
            style='rounded,filled',
            fillcolor='white',
            fontname='Arial',
            margin='0.3,0.2'
        )
        
        # Edge styles
        dot.attr('edge',
            color='#1E88E5',
            penwidth='2'
        )
        
        # Add treatment and outcome nodes
        treatment = st.session_state.treatment_input
        outcome = st.session_state.outcome_input
        dot.node(treatment, treatment)
        dot.node(outcome, outcome)
        
        # Add IV nodes and edges
        for iv in ivs:
            if isinstance(iv, (list, tuple)) and len(iv) >= 2:
                iv_name = str(iv[0]).strip()
                explanation = str(iv[1]).strip()
                validity = float(iv[2]) if len(iv) > 2 else 0.5
                
                # Add IV node
                dot.node(iv_name, iv_name)
                
                # Add edge (only to treatment)
                dot.edge(iv_name, treatment)
                
                # Format validity level
                if validity > 0.7:
                    validity_level = "high"
                    validity_color = "#27ae60"
                elif validity > 0.4:
                    validity_level = "medium"
                    validity_color = "#f39c12"
                else:
                    validity_level = "low"
                    validity_color = "#e74c3c"
                
                # Display IV information
                with st.expander(f"🎯 {iv_name} (Instrumental Variable)", expanded=True):
                    st.markdown(f"""
                        <div style='color: {validity_color}; font-weight: bold; margin-bottom: 10px;'>
                            Validity Level: {validity_level.title()}
                            <br>
                            Instrument Strength: {validity:.2f}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Justification:** {explanation}")
                    st.markdown("""
                        **IV Assumptions:**
                        1. ✓ Affects treatment
                        2. ✓ No direct effect on outcome
                        3. ✓ Independent of confounders
                    """)
        
        # Display the visualization
        st.graphviz_chart(dot)
        
        # Add explanation
        with st.expander("🔍 Understanding Instrumental Variables", expanded=True):
            st.markdown("""
            ### What are Instrumental Variables?
            IVs help estimate causal effects when there are unmeasured confounders:
            1. They influence the treatment
            2. They only affect the outcome through the treatment
            3. They are independent of confounders
            
            ### How to Use IVs
            - Use them in IV regression
            - Test instrument strength
            - Validate exclusion restriction
            - Consider multiple instruments if available
            """)
        
    except Exception as e:
        st.error(f"Error formatting IVs: {str(e)}")
        return None

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error(OPENAI_API_KEY_ERROR)
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

    # Initialize session state variables if they don't exist
    if 'suggested_treatment' not in st.session_state:
        st.session_state.suggested_treatment = ""
    if 'suggested_outcome' not in st.session_state:
        st.session_state.suggested_outcome = ""
    if 'factors_input' not in st.session_state:
        st.session_state.factors_input = ""
    if 'treatment_input' not in st.session_state:
        st.session_state.treatment_input = ""
    if 'outcome_input' not in st.session_state:
        st.session_state.outcome_input = ""

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
        
        all_factors_str = st.text_area(
            "📝 Enter all relevant factors (comma-separated):", 
            value=st.session_state.factors_input,
            help=factors_help,
            key="factors_area"
        )
        
        # Update session state for factors
        if all_factors_str != st.session_state.factors_input:
            st.session_state.factors_input = all_factors_str
        
        all_factors = [factor.strip() for factor in all_factors_str.split(',') if factor.strip()]

        # Add a button to suggest variables
        if st.button("🎯 Suggest Treatment and Outcome Variables"):
            if all_factors:
                with st.spinner("Analyzing factors to suggest variables..."):
                    try:
                        suggested_treatment, suggested_outcome = suggest_variables_from_factors(all_factors, openai_api_key)
                        if suggested_treatment and suggested_outcome:
                            st.session_state.suggested_treatment = suggested_treatment
                            st.session_state.suggested_outcome = suggested_outcome
                            st.session_state.treatment_input = suggested_treatment
                            st.session_state.outcome_input = suggested_outcome
                            st.success(f"Variables suggested! Treatment: {suggested_treatment}, Outcome: {suggested_outcome}")
                        else:
                            st.warning("Could not generate suggestions. Please check your input factors.")
                    except Exception as e:
                        st.error(f"Error during suggestion: {str(e)}")
            else:
                st.warning(MISSING_FACTORS_ERROR)

        # Treatment input with session state
        treatment = st.text_input(
            "🎯 Enter the treatment variable:",
            value=st.session_state.treatment_input,
            help="The variable whose effect you want to study.",
            key="treatment_field"
        )
        
        # Update session state for treatment
        if treatment != st.session_state.treatment_input:
            st.session_state.treatment_input = treatment

        # Outcome input with session state
        outcome = st.text_input(
            "🎯 Enter the outcome variable:",
            value=st.session_state.outcome_input,
            help="The variable you want to measure the effect on.",
            key="outcome_field"
        )
        
        # Update session state for outcome
        if outcome != st.session_state.outcome_input:
            st.session_state.outcome_input = outcome

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
                if all_factors and treatment and outcome:
                    with st.spinner("Analyzing potential confounding variables..."):
                        try:
                            suggested_confounders = suggest_confounders_from_factors(
                                treatment, outcome, all_factors, openai_api_key
                            )
                            if suggested_confounders:
                                st.subheader("Potential Confounding Variables")
                                formatted_confounders = format_confounder_output(suggested_confounders)
                                st.markdown(formatted_confounders)
                        except Exception as e:
                            st.error(f"Error during confounders suggestion: {str(e)}")
                else:
                    st.warning(MISSING_VARIABLES_ERROR)

            if st.button("Suggest Pair-wise Relationships (DAG)"):
                if all_factors and treatment and outcome:
                    if not openai_api_key:
                        st.error("Please set your OpenAI API key in the environment variables.")
                    else:
                        with st.spinner("Analyzing potential relationships between variables..."):
                            try:
                                suggested_relationships = suggest_relationships_from_factors(
                                    treatment, outcome, all_factors, openai_api_key
                                )
                                if suggested_relationships:
                                    st.subheader("Suggested Pair-wise Relationships (Potential DAG Edges)")
                                    st.success("Successfully identified relationships between variables!")
                                    formatted_relationships = format_relationship_output(suggested_relationships)
                                    if formatted_relationships:
                                        st.markdown(formatted_relationships)
                                    else:
                                        st.warning("No clear relationships were identified. Try adjusting your input variables or adding more context.")
                                else:
                                    st.warning("No relationships could be identified. Please check your input variables and try again.")
                            except Exception as e:
                                st.error(f"An error occurred while analyzing relationships: {str(e)}")
                                st.info("Try simplifying your input or checking for any special characters in variable names.")
                else:
                    st.warning(MISSING_VARIABLES_ERROR)

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
                if all_factors and treatment and outcome:
                    if not openai_api_key:
                        st.error("Please set your OpenAI API key in the environment variables.")
                    else:
                        with st.spinner("Analyzing variables to identify backdoor adjustment set..."):
                            try:
                                suggested_backdoor = suggest_backdoor_from_factors(
                                    treatment, outcome, all_factors, openai_api_key
                                )
                                if suggested_backdoor:
                                    st.success("Successfully identified backdoor adjustment set!")
                                    formatted_backdoor = format_backdoor_set(suggested_backdoor)
                                    st.markdown(formatted_backdoor)
                                else:
                                    st.warning("No clear backdoor adjustment set could be identified. Please check your input variables.")
                            except Exception as e:
                                st.error(f"Error during backdoor set suggestion: {str(e)}")
                else:
                    st.warning(MISSING_VARIABLES_ERROR)

            if st.button("Suggest Mediator Set"):
                if all_factors and treatment and outcome:
                    if not openai_api_key:
                        st.error("Please set your OpenAI API key in the environment variables.")
                    else:
                        with st.spinner("Analyzing variables to identify mediators..."):
                            try:
                                suggested_mediators = suggest_mediator_from_factors(
                                    treatment, outcome, all_factors, openai_api_key
                                )
                                if suggested_mediators:
                                    st.success("Successfully identified mediator variables!")
                                    format_mediator_output(suggested_mediators)
                                else:
                                    st.warning("No clear mediator variables could be identified. Please check your input variables.")
                            except Exception as e:
                                st.error(f"Error during mediator suggestion: {str(e)}")
                else:
                    st.warning(MISSING_VARIABLES_ERROR)

            if st.button("Suggest Instrumental Variables (IVs)"):
                if all_factors and treatment and outcome:
                    if not openai_api_key:
                        st.error("Please set your OpenAI API key in the environment variables.")
                    else:
                        with st.spinner("Analyzing variables to identify instrumental variables..."):
                            try:
                                suggested_ivs = suggest_iv_from_factors(
                                    treatment, outcome, all_factors, openai_api_key
                                )
                                if suggested_ivs:
                                    st.success("Successfully identified instrumental variables!")
                                    format_iv_output(suggested_ivs)
                                else:
                                    st.warning("No clear instrumental variables could be identified. Please check your input variables.")
                            except Exception as e:
                                st.error(f"Error during IV suggestion: {str(e)}")
                else:
                    st.warning(MISSING_VARIABLES_ERROR)

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