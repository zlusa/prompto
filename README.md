# Prompt Wizard

A Streamlit application for analyzing and optimizing prompts using Google's Gemini model. The application supports both text and image inputs, providing detailed analysis and generating optimized prompts.

## Features

- Text prompt analysis and optimization
- Image analysis with detailed visual descriptions
- Character consistency options for image prompts
- Additional context input for tailored prompts
- Clean prompt display with easy copy functionality

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PromptWizard.git
cd PromptWizard
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Running the Application Locally

Start the Streamlit application:
```bash
streamlit run prompt_analyzer_app.py
```

The application will open in your default web browser.

## Deployment on Streamlit Cloud

1. Push your code to GitHub if you haven't already
2. Visit [Streamlit Cloud](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Set up the deployment:
   - Main file path: `prompt_analyzer_app.py`
   - Python version: 3.12
6. Add your secrets in Streamlit Cloud:
   - Go to "Advanced settings" > "Secrets"
   - Add your environment variables:
     ```toml
     GOOGLE_API_KEY = "your_api_key_here"
     ```
7. Click "Deploy!"

Your app will be available at `https://your-app-name.streamlit.app`

## Usage

### Text Analysis
1. Select the "Text" tab
2. Enter your prompt in the text area
3. Click "Analyze" to get detailed feedback and optimized versions

### Image Analysis
1. Select the "Image" tab
2. Upload an image
3. (Optional) Add additional context or requirements
4. Check the "Contains Character" box if your image includes a character
5. Click "Analyze" to get detailed visual analysis and optimized prompts

## Contributing

Feel free to submit issues and enhancement requests!

