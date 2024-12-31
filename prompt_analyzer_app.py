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

def analyze_input(user_input, input_type="text"):
    """Let LLM analyze the input and determine role and instructions"""
    
    gemini_analysis = None  # Initialize gemini_analysis
    
    if input_type.lower() == "image":
        try:
            image_data = json.loads(user_input)
            
            # First get Gemini's analysis of the image
            image_bytes = base64.b64decode(image_data['image_base64'])
            gemini_analysis = analyze_image_with_gemini(image_bytes, image_data['additional_details'])
            
            if not gemini_analysis:
                st.error("Failed to get Gemini analysis")
                return None
            
            # Create prompt for the second phase of analysis
            analysis_content = f"""Based on this detailed visual analysis, help create effective prompts:

            VISUAL ANALYSIS:
            {gemini_analysis}

            ADDITIONAL CONTEXT:
            {image_data['additional_details']}

            Create a structured analysis that can help generate effective prompts for recreating or describing this image."""
        except json.JSONDecodeError:
            analysis_content = user_input
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    else:
        analysis_content = user_input
    
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
            "content": f"""Analyze this {input_type} input and provide a detailed visual analysis:

            {analysis_content}"""
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
        st.error(f"Error analyzing input: {str(e)}")
        return None

def generate_optimized_prompt(analysis, input_type="text", additional_context=""):
    """Generate optimized prompt based on analysis and context"""
    
    context_specific = f"""
    Consider these specific requirements:
    - Purpose: {additional_context}
    - Style elements from the analysis: {analysis.get('visual_analysis', {}).get('style_analysis', '')}
    - Mood elements from the analysis: {analysis.get('visual_analysis', {}).get('mood_and_tone', '')}
    """
    
    optimization_prompt = [
        {
            "role": "system",
            "content": """You are an expert prompt optimizer who excels at creating detailed, context-aware prompts.
            Consider:
            - Specific requirements and context
            - Visual style and elements
            - Mood and atmosphere
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
            Role: {analysis['detected_role']}
            Task: {analysis['task_description']}
            Base Instruction: {analysis['base_instruction']}
            {context_specific if input_type.lower() == "image" else ""}"""
        }
    ]
    
    try:
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
                            if 'visual_analysis' in analysis:
                                visual = analysis['visual_analysis']
                                for key, value in visual.items():
                                    st.write(f"**{key.replace('_', ' ').title()}:**")
                                    st.info(value)
                    
                    with col2:
                        with st.expander("💭 Reasoning", expanded=True):
                            st.info(analysis['reasoning'])
                
                # Prompts section with full width
                st.subheader("✨ Generated Prompts")
                
                # Optimized prompt first
                with st.expander("Optimized Prompt", expanded=True):
                    with st.spinner("Generating optimized prompt..."):
                        optimized_prompt = generate_optimized_prompt(analysis, input_type.lower(), additional_details_str)
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
                
                # Create two columns for the prompts
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