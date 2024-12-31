# Standard library imports
import os
import io
import json
import base64

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import PIL.Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pyperclip

# Load environment variables
load_dotenv()

# Initialize session state for additional details
if 'additional_details_str' not in st.session_state:
    st.session_state.additional_details_str = ""

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def call_api(messages):
    """Call the Gemini API with the given messages."""
    try:
        # Convert messages to a single prompt
        prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Generate response
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")

def analyze_image_with_gemini(image_bytes, additional_context=""):
    """Analyze image using Gemini model"""
    try:
        # Convert bytes to PIL Image
        image = PIL.Image.open(io.BytesIO(image_bytes))
        
        # Create the prompt with context-aware analysis
        prompt = f"""Analyze this image in great detail and consider how it relates to: "{additional_context}"

        1. Visual Analysis:
        - What you see in the image (main subjects, objects, elements)
        - The style and artistic elements
        - Colors, lighting, and overall visual feel
        - Any notable expressions, emotions, or mood
        - Composition and layout
        - Any unique or distinctive features

        2. Context Integration:
        - How does this image relate to "{additional_context}"?
        - What elements could be adapted or enhanced for this context?
        - What mood or atmosphere would work well with this context?
        
        3. Creative Possibilities:
        - How could this image be modified or adapted for the given context?
        - What additional elements would enhance the image for this purpose?
        - What style adjustments would make it more suitable?

        Provide a detailed analysis that considers both the image itself and how it could be optimized for: {additional_context if additional_context else 'general use'}"""

        # Create content parts list with text and image
        content = [
            prompt,  # Text part
            image   # Image part
        ]
        
        response = model.generate_content(content, stream=False)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing image with Gemini: {str(e)}")
        return None

def analyze_text(text_input):
    """Analyze text input and generate optimized prompts"""
    analysis_prompt = [
        {
            "role": "system",
            "content": """You are an expert prompt engineer who excels at analyzing user requests and determining the most appropriate expert role. Your task is to:
            1. Analyze the user's input to understand their intent and required expertise
            2. Determine the most appropriate expert role(s) for this task
            3. Create prompts that embody the expertise and perspective of that role

            For example:
            - For "Write a blog post about AI trends": Role = Professional Writer & SEO Expert
            - For "Help me understand quadratic equations": Role = Math Teacher & Educational Expert
            - For "Create a marketing campaign": Role = Marketing Strategist & Brand Expert

            Provide your analysis in the following JSON format:
            {
                "detected_roles": {
                    "primary_role": "The main expert role needed",
                    "secondary_roles": ["Additional expert roles that would be valuable"],
                    "role_justification": "Explanation of why these roles are most appropriate"
                },
                "task_analysis": {
                    "core_objective": "The main goal to be achieved",
                    "key_requirements": ["List of critical requirements"],
                    "target_audience": "Who this is intended for",
                    "success_criteria": ["What defines success for this task"]
                },
                "expert_perspective": {
                    "approach": "How the expert would approach this task",
                    "key_considerations": ["Important factors the expert would consider"],
                    "professional_tips": ["Expert tips and best practices"],
                    "common_pitfalls": ["What to avoid, from an expert's perspective"]
                },
                "role_specific_prompts": {
                    "expert_prompt": "A detailed prompt written from the expert's perspective",
                    "step_by_step": ["Detailed steps the expert would take"],
                    "advanced_techniques": ["Specialized methods or approaches"],
                    "quality_checks": ["How to verify the quality of the output"]
                },
                "optimized_prompts": {
                    "detailed": "A comprehensive, detailed prompt",
                    "concise": "A shorter, focused version",
                    "creative": "An artistic or imaginative take"
                },
                "additional_suggestions": ["List of tips or modifications"],
                "reasoning": "Explanation of the analysis and prompt design choices"
            }"""
        },
        {
            "role": "user",
            "content": f"""Analyze this request and determine the most appropriate expert role(s):

            {text_input}

            Provide a detailed analysis from the expert's perspective and create optimized prompts that embody their expertise."""
        }
    ]
    
    try:
        response = call_api(analysis_prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

def analyze_code(code_input):
    """Analyze code input and generate optimized prompts"""
    analysis_prompt = [
        {
            "role": "system",
            "content": """You are an expert code analyst and prompt engineer who excels at:
            1. Understanding code structure and purpose
            2. Identifying programming patterns and best practices
            3. Creating effective prompts for code-related tasks
            4. Optimizing technical communication

            Provide your analysis in the following JSON format:
            {
                "detected_language": "Programming language identified",
                "code_analysis": {
                    "purpose": "Main purpose or functionality of the code",
                    "structure": "Analysis of code structure and organization",
                    "patterns": "Identified programming patterns or paradigms",
                    "technical_requirements": "Technical specifications or dependencies"
                },
                "task_description": "Clear description of the code-related task",
                "base_instruction": "Core technical instruction",
                "example_prompts": ["List of 3 alternative prompts for code-related tasks"],
                "reasoning": "Technical explanation of the analysis and prompt choices"
            }"""
        },
        {
            "role": "user",
            "content": f"""Analyze this code and create optimized prompts:

            {code_input}"""
        }
    ]
    
    try:
        response = call_api(analysis_prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        st.error(f"Error analyzing code: {str(e)}")
        return None

def analyze_image(image_data):
    """Analyze image input and generate optimized prompts"""
    try:
        # First get Gemini's analysis of the image
        image_bytes = base64.b64decode(image_data['image_base64'])
        gemini_analysis = analyze_image_with_gemini(image_bytes, image_data['additional_details'])
        
        if not gemini_analysis:
            st.error("Failed to get Gemini analysis")
            return None
        
        # Store additional details in session state
        st.session_state.additional_details_str = image_data['additional_details']
        
        # Create prompt for the second phase of analysis
        analysis_prompt = [
            {
                "role": "system",
                "content": """You are an expert visual analyst and prompt engineer who excels at:
                1. Providing extremely detailed visual descriptions
                2. Identifying key artistic and technical elements
                3. Understanding style and composition
                4. Creating effective prompts based on visual analysis

                Provide your analysis in the following JSON format:
                {
                    "detected_role": "Visual Analysis Expert",
                    "visual_analysis": {
                        "subject_description": "Detailed description of the main subject",
                        "style_analysis": "Analysis of artistic style and technique",
                        "composition_details": "Analysis of composition and layout",
                        "mood_and_tone": "Description of emotional quality and atmosphere",
                        "technical_aspects": "Analysis of technical image qualities"
                    },
                    "task_description": "Detailed description of what needs to be done",
                    "base_instruction": "Step-by-step approach to recreate or understand this image",
                    "example_prompts": ["List of 3 alternative prompts incorporating the analysis"],
                    "reasoning": "Explanation of the analysis and prompt choices"
                }"""
            },
            {
                "role": "user",
                "content": f"""Based on this detailed visual analysis, help create effective prompts:

                VISUAL ANALYSIS:
                {gemini_analysis}

                ADDITIONAL CONTEXT:
                {image_data['additional_details']}

                Create a structured analysis that can help generate effective prompts for recreating or describing this image."""
            }
        ]
        
        response = call_api(analysis_prompt)
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result['gemini_analysis'] = gemini_analysis
            return result
        return None
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def analyze_input(user_input, input_type="text"):
    """Route the input to the appropriate analyzer based on type"""
    if input_type.lower() == "text":
        return analyze_text(user_input)
    elif input_type.lower() == "code":
        return analyze_code(user_input)
    elif input_type.lower() == "image":
        try:
            image_data = json.loads(user_input)
            return analyze_image(image_data)
        except json.JSONDecodeError:
            st.error("Invalid image data format")
            return None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    else:
        st.error(f"Unsupported input type: {input_type}")
        return None

def generate_optimized_prompt(analysis, input_type="text"):
    """Generate optimized prompt based on analysis and context"""
    
    if input_type.lower() == "image":
        context_specific = f"""
        Consider these specific requirements:
        - Purpose: {st.session_state.additional_details_str}
        - Style elements from the analysis: {analysis.get('visual_analysis', {}).get('style_analysis', '')}
        - Mood elements from the analysis: {analysis.get('visual_analysis', {}).get('mood_and_tone', '')}
        """
    elif input_type.lower() == "text":
        context_specific = f"""
        Consider these specific requirements:
        - Core Theme: {analysis.get('core_theme', '')}
        - Intent: {analysis.get('prompt_analysis', {}).get('intent', '')}
        - Key Elements: {', '.join(analysis.get('prompt_analysis', {}).get('key_elements', []))}
        - Style Suggestions: {', '.join(analysis.get('prompt_analysis', {}).get('style_suggestions', []))}
        """
    else:  # code
        context_specific = f"""
        Consider these specific requirements:
        - Language: {analysis.get('detected_language', '')}
        - Purpose: {analysis.get('code_analysis', {}).get('purpose', '')}
        - Technical requirements: {analysis.get('code_analysis', {}).get('technical_requirements', '')}
        """
    
    optimization_prompt = [
        {
            "role": "system",
            "content": """You are an expert prompt optimizer who excels at creating detailed, context-aware prompts.
            Consider:
            - Specific requirements and context
            - Style and tone elements
            - Technical specifications
            - Practical applications
            
            Create prompts that are:
            1. Highly detailed and specific
            2. Contextually relevant
            3. Technically accurate
            4. Practically applicable"""
        },
        {
            "role": "user",
            "content": f"""Create an optimized prompt based on this analysis and context:
            {context_specific}"""
        }
    ]
    
    try:
        if input_type.lower() == "text":
            # For text input, return the detailed prompt from the analysis
            return analysis.get('optimized_prompts', {}).get('detailed', '')
        return call_api(optimization_prompt)
    except Exception as e:
        st.error(f"Error generating optimized prompt: {str(e)}")
        return None

# Streamlit UI
st.title("🧙‍♂️ AI Prompt Wizard")
st.write("Enter your input, and I'll determine the best way to handle it!")

# Input section
input_type = st.radio("Input Type:", ["Text", "Image", "Code"])
user_input = ""

if input_type == "Text":
    user_input = st.text_area("Enter your text:", height=150)
elif input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    
    # Add more structured context input
    with st.expander("🎯 Image Context & Requirements", expanded=True):
        context_purpose = st.text_input("Purpose/Theme (e.g., 'New Year Celebration'):", "")
        
        # Add character consistency section
        has_character = st.checkbox("This image contains a character (person, animal, creature, etc.)")
        if has_character:
            st.write("Character Details (for consistency):")
            char_cols = st.columns(2)
            with char_cols[0]:
                char_type = st.selectbox("Character Type:", [
                    "Person", "Animal", "Creature", "Cartoon Character", "Other"
                ])
                char_name = st.text_input("Character Name (optional):", "")
                char_gender = st.selectbox("Gender (if applicable):", [
                    "Not Specified", "Male", "Female", "Other"
                ])
            with char_cols[1]:
                char_age = st.text_input("Age/Stage (e.g., young, adult, elderly):", "")
                char_distinct = st.text_area("Distinctive Features:", 
                    placeholder="Key features to maintain consistency (e.g., blue eyes, curly hair, wings, etc.)",
                    height=100)
        
        context_style = st.selectbox("Desired Style:", [
            "Keep Original", "Festive", "Professional", "Artistic", "Minimalist", 
            "Vintage", "Modern", "Playful", "Elegant", "Custom"
        ])
        if context_style == "Custom":
            context_style = st.text_input("Specify custom style:", "")
        
        context_mood = st.selectbox("Desired Mood:", [
            "Keep Original", "Celebratory", "Energetic", "Calm", "Serious", 
            "Whimsical", "Dramatic", "Peaceful", "Professional", "Custom"
        ])
        if context_mood == "Custom":
            context_mood = st.text_input("Specify custom mood:", "")
        
        additional_notes = st.text_area("Additional Notes or Requirements:", height=100)
        
        # Combine all context
        additional_details = {
            "purpose": context_purpose,
            "style": context_style if context_style != "Keep Original" else "",
            "mood": context_mood if context_mood != "Keep Original" else "",
            "notes": additional_notes
        }
        
        # Add character details if present
        if has_character:
            additional_details["character"] = {
                "type": char_type,
                "name": char_name if char_name else "Unspecified",
                "gender": char_gender if char_gender != "Not Specified" else "",
                "age": char_age,
                "distinctive_features": char_distinct
            }
        
        additional_details_str = "\n".join([f"{k}: {v}" for k, v in additional_details.items() if v and k != "character"])
        if "character" in additional_details:
            char_details = additional_details["character"]
            additional_details_str += "\nCharacter Details:\n" + "\n".join([f"- {k}: {v}" for k, v in char_details.items() if v])
        
        if uploaded_file:
            import base64
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Combine image and additional details
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            user_input = {
                "image_base64": image_base64,
                "additional_details": additional_details_str,
                "filename": uploaded_file.name,
                "file_type": uploaded_file.type
            }
            user_input = json.dumps(user_input)  # Convert to JSON string for processing
else:  # Code
    user_input = st.text_area("Enter your code:", height=150)
    
# Analysis button
if st.button("Analyze & Generate Prompts"):
    if user_input:
        with st.spinner("Analyzing input..."):
            # Initialize gemini_analysis at the correct scope
            gemini_analysis = None
            
            # Get analysis
            analysis = analyze_input(user_input, input_type.lower())
            
            # If it's an image, get the Gemini analysis first
            if input_type.lower() == "image" and analysis:
                try:
                    image_data = json.loads(user_input)
                    image_bytes = base64.b64decode(image_data['image_base64'])
                    gemini_analysis = analyze_image_with_gemini(image_bytes, image_data['additional_details'])
                except Exception as e:
                    st.error(f"Error getting Gemini analysis: {str(e)}")
            
            if analysis:
                # Display analysis
                st.subheader("📊 Analysis")
                
                # Use a container for better spacing
                with st.container():
                    col1, col2 = st.columns([6, 4])
                    
                    with col1:
                        if input_type.lower() == "image" and gemini_analysis:
                            with st.expander("🔍 Gemini's Visual Analysis", expanded=True):
                                st.markdown(gemini_analysis)
                        
                        with st.expander("🎨 Detailed Analysis", expanded=True):
                            if input_type.lower() == "image" and 'visual_analysis' in analysis:
                                visual = analysis['visual_analysis']
                                for key, value in visual.items():
                                    st.write(f"**{key.replace('_', ' ').title()}:**")
                                    st.info(value)
                            elif input_type.lower() == "text":
                                # Display detected roles
                                st.write("**🎭 Expert Roles:**")
                                roles = analysis.get('detected_roles', {})
                                st.write("*Primary Role:*")
                                st.success(roles.get('primary_role', ''))
                                
                                st.write("*Secondary Roles:*")
                                for role in roles.get('secondary_roles', []):
                                    st.info(f"• {role}")
                                
                                st.write("*Role Justification:*")
                                st.info(roles.get('role_justification', ''))
                                
                                # Display task analysis
                                st.write("**📋 Task Analysis:**")
                                task = analysis.get('task_analysis', {})
                                st.write("*Core Objective:*")
                                st.success(task.get('core_objective', ''))
                                
                                st.write("*Key Requirements:*")
                                for req in task.get('key_requirements', []):
                                    st.info(f"• {req}")
                                
                                st.write("*Target Audience:*")
                                st.warning(task.get('target_audience', ''))
                                
                                st.write("*Success Criteria:*")
                                for crit in task.get('success_criteria', []):
                                    st.success(f"• {crit}")
                    
                    with col2:
                        with st.expander("💭 Expert Perspective", expanded=True):
                            if input_type.lower() == "text":
                                expert = analysis.get('expert_perspective', {})
                                st.write("**Approach:**")
                                st.info(expert.get('approach', ''))
                                
                                st.write("**Key Considerations:**")
                                for cons in expert.get('key_considerations', []):
                                    st.success(f"• {cons}")
                                
                                st.write("**Professional Tips:**")
                                for tip in expert.get('professional_tips', []):
                                    st.info(f"• {tip}")
                                
                                st.write("**Common Pitfalls:**")
                                for pitfall in expert.get('common_pitfalls', []):
                                    st.error(f"• {pitfall}")
                            else:
                                st.info(analysis['reasoning'])
                        
                        if input_type.lower() == "text":
                            with st.expander("🎯 Additional Suggestions", expanded=True):
                                for suggestion in analysis.get('additional_suggestions', []):
                                    st.success(f"• {suggestion}")
                
                # Prompts section with full width
                st.subheader("✨ Generated Prompts")
                
                if input_type.lower() == "text":
                    # Display expert-specific prompts first
                    role_prompts = analysis.get('role_specific_prompts', {})
                    
                    with st.expander("👨‍💼 Expert's Detailed Prompt", expanded=True):
                        expert_prompt = role_prompts.get('expert_prompt', '')
                        st.success(expert_prompt)
                        if st.button("📋 Copy Expert", key="copy_expert"):
                            st.code(expert_prompt)
                            st.success("✅ Expert prompt copied!")
                    
                    with st.expander("📝 Expert's Step-by-Step Guide", expanded=True):
                        st.write("**Steps:**")
                        for step in role_prompts.get('step_by_step', []):
                            st.info(f"• {step}")
                    
                    with st.expander("🎓 Advanced Techniques", expanded=True):
                        st.write("**Specialized Methods:**")
                        for technique in role_prompts.get('advanced_techniques', []):
                            st.success(f"• {technique}")
                    
                    with st.expander("✅ Quality Checks", expanded=True):
                        st.write("**Verification Steps:**")
                        for check in role_prompts.get('quality_checks', []):
                            st.warning(f"• {check}")
                    
                    st.divider()
                    
                    # Display the standard optimized prompts
                    optimized_prompts = analysis.get('optimized_prompts', {})
                    
                    with st.expander("Detailed Prompt", expanded=True):
                        detailed = optimized_prompts.get('detailed', '')
                        st.success(detailed)
                        if st.button("📋 Copy Detailed", key="copy_detailed"):
                            st.code(detailed)
                            st.success("✅ Detailed prompt copied!")
                    
                    with st.expander("Concise Prompt", expanded=True):
                        concise = optimized_prompts.get('concise', '')
                        st.success(concise)
                        if st.button("📋 Copy Concise", key="copy_concise"):
                            st.code(concise)
                            st.success("✅ Concise prompt copied!")
                    
                    with st.expander("Creative Prompt", expanded=True):
                        creative = optimized_prompts.get('creative', '')
                        st.success(creative)
                        if st.button("📋 Copy Creative", key="copy_creative"):
                            st.code(creative)
                            st.success("✅ Creative prompt copied!")
                else:
                    # Original prompt display for image and code
                    with st.expander("Optimized Prompt", expanded=True):
                        with st.spinner("Generating optimized prompt..."):
                            optimized_prompt = generate_optimized_prompt(analysis, input_type.lower())
                            if optimized_prompt:
                                st.success(optimized_prompt)
                    
                    # Alternative prompts below
                    for i, prompt in enumerate(analysis['example_prompts'], 1):
                        with st.expander(f"Alternative Prompt {i}", expanded=True):
                            st.text_area(f"Alternative {i}", value=prompt, height=150, key=f"alt_prompt_{i}", disabled=True)
                
                # Clean Prompts Display Section
                st.divider()
                st.subheader("🎯 Clean Prompts")
                st.write("Copy-ready prompts without explanations:")
                
                if input_type.lower() == "text":
                    # Create three columns for text prompts
                    clean_col1, clean_col2, clean_col3 = st.columns([1, 1, 1])
                    
                    with clean_col1:
                        st.markdown("**✨ Detailed:**")
                        st.text_area("", value=optimized_prompts.get('detailed', ''), height=150, key="clean_detailed", disabled=True)
                    
                    with clean_col2:
                        st.markdown("**💫 Concise:**")
                        st.text_area("", value=optimized_prompts.get('concise', ''), height=150, key="clean_concise", disabled=True)
                    
                    with clean_col3:
                        st.markdown("**🎨 Creative:**")
                        st.text_area("", value=optimized_prompts.get('creative', ''), height=150, key="clean_creative", disabled=True)
                else:
                    # Original two-column layout for image and code
                    prompt_col1, prompt_col2 = st.columns([1, 1])
                    
                    with prompt_col1:
                        st.markdown("**✨ Optimized Prompt:**")
                        st.text_area("", value=optimized_prompt, height=150, key="clean_opt", disabled=True)
                        if st.button("📋 Copy", key="copy_opt"):
                            st.code(optimized_prompt)
                            st.success("✅ Optimized prompt copied!")
                    
                    with prompt_col2:
                        for i, prompt in enumerate(analysis['example_prompts'], 1):
                            st.markdown(f"**Alternative {i}:**")
                            st.text_area("", value=prompt, height=150, key=f"clean_alt_{i}", disabled=True)
                            if st.button(f"📋 Copy", key=f"copy_alt_{i}"):
                                st.code(prompt)
                                st.success(f"✅ Alternative {i} copied!")
                
                # Remove the old copy buttons section since we have the clean copy functionality above
                st.divider()
                
                # Save Results button
                if st.button("💾 Save Results", key="save_results", use_container_width=True):
                    # Create results directory if it doesn't exist
                    os.makedirs("results", exist_ok=True)
                    
                    # Save results to file
                    results = {
                        "analysis": analysis,
                        "optimized_prompt": optimized_prompt,
                        "input_type": input_type,
                        "timestamp": str(pd.Timestamp.now())
                    }
                    
                    filename = f"results/prompt_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    st.success(f"Results saved to {filename}")
    else:
        st.warning("Please enter some input first!")

# Add some helpful tips
with st.sidebar:
    st.subheader("💡 Tips")
    st.write("""
    - Be specific in your input
    - For code, include context/comments
    - For images, clear images work best
    - You can save results for later use
    """)
    
    st.subheader("📚 Recent Analyses")
    # Show recent analyses from saved files
    results_dir = "results"
    if os.path.exists(results_dir):
        files = sorted(os.listdir(results_dir), reverse=True)[:5]
        for file in files:
            try:
                with st.expander(file):
                    with open(os.path.join(results_dir, file), "r") as f:
                        data = json.load(f)
                    st.write(f"Type: {data.get('input_type', 'N/A')}")
                    st.write(f"Role: {data.get('analysis', {}).get('detected_role', 'N/A')}")
            except Exception as e:
                st.error(f"Error reading {file}: {str(e)}") 