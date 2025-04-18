import os
import streamlit as st
import pandas as pd
import anthropic
import google.generativeai as genai
import PyPDF2
import docx2txt
import re
import json
from pathlib import Path
import tempfile
import shutil
import traceback

# Handle different versions of OpenAI package
try:
    # For newer versions of the OpenAI package (>=1.0.0)
    from openai import OpenAI
    OPENAI_NEW_VERSION = True
except ImportError:
    # For older versions of the OpenAI package (<1.0.0)
    import openai
    OPENAI_NEW_VERSION = False

st.set_page_config(page_title="Resume Shortlister", layout="wide")

# Function to extract text from various file types
def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                return text
        
        elif file_extension == '.docx':
            text = docx2txt.process(file_path)
            return text
        
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        else:
            return f"Unsupported file format: {file_extension}"
    
    except Exception as e:
        return f"Error processing file {file_path}: {str(e)}"

# Function to analyze resume using Claude API
def analyze_with_claude(resume_text, requirements, api_key):
    # Make sure to explicitly set api_key in the client constructor
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""
    You are a professional resume analyzer. Please analyze the following resume against these job requirements:
    
    REQUIREMENTS:
    {requirements}
    
    RESUME:
    {resume_text}
    
    Please provide your analysis in JSON format with the following structure:
    {{
        "match_score": (float between 0 and 1),
        "experience_match": (boolean),
        "skills_match": (boolean),
        "location_match": (boolean),
        "project_experience_match": (boolean),
        "has_github": (boolean),
        "has_tech_publications": (boolean),
        "has_code_contributions": (boolean),
        "key_strengths": (list of strings),
        "key_weaknesses": (list of strings),
        "rationale": (string with detailed explanation)
    }}
    
    Focus on quantifiable metrics and concrete evidence from the resume. Be fair but thorough in your assessment.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0,
            system="You are a professional resume analyzer who provides accurate, fair assessments in JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        analysis_text = response.content[0].text
        # Find JSON in the response
        match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
        if match:
            analysis_text = match.group(1)
        else:
            # Try to find JSON without code block
            match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if match:
                analysis_text = match.group(1)
            else:
                # Last resort, try to use the entire text
                analysis_text = analysis_text
        
        try:
            return json.loads(analysis_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure with error info
            return {
                "error": "Failed to parse LLM response as JSON",
                "match_score": 0.0,
                "experience_match": False,
                "skills_match": False,
                "location_match": False,
                "project_experience_match": False,
                "has_github": False,
                "has_tech_publications": False,
                "has_code_contributions": False,
                "key_strengths": [],
                "key_weaknesses": ["Could not analyze resume properly"],
                "rationale": f"Error processing resume: Failed to parse response as JSON. Raw response: {analysis_text[:500]}..."
            }
    
    except Exception as e:
        # Return a structured error with traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {
            "error": error_msg,
            "match_score": 0.0,
            "experience_match": False,
            "skills_match": False,
            "location_match": False,
            "project_experience_match": False,
            "has_github": False,
            "has_tech_publications": False,
            "has_code_contributions": False,
            "key_strengths": [],
            "key_weaknesses": ["Could not analyze resume properly"],
            "rationale": f"Error processing resume: {error_msg}. Stack trace: {stack_trace}"
        }

# Function to analyze resume using OpenAI API
def analyze_with_openai(resume_text, requirements, api_key):
    # Set the API key
    if OPENAI_NEW_VERSION:
        client = OpenAI(api_key=api_key)
    else:
        openai.api_key = api_key
    
    prompt = f"""
    You are a professional resume analyzer. Please analyze the following resume against these job requirements:
    
    REQUIREMENTS:
    {requirements}
    
    RESUME:
    {resume_text}
    
    Please provide your analysis in JSON format with the following structure:
    {{
        "match_score": (float between 0 and 1),
        "experience_match": (boolean),
        "skills_match": (boolean),
        "location_match": (boolean),
        "project_experience_match": (boolean),
        "has_github": (boolean),
        "has_tech_publications": (boolean),
        "has_code_contributions": (boolean),
        "key_strengths": (list of strings),
        "key_weaknesses": (list of strings),
        "rationale": (string with detailed explanation)
    }}
    
    Focus on quantifiable metrics and concrete evidence from the resume. Be fair but thorough in your assessment.
    """
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a professional resume analyzer who provides accurate, fair assessments in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
        else:
            # Using older OpenAI API format
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a professional resume analyzer who provides accurate, fair assessments in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message['content']
        
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure with error info
            return {
                "error": "Failed to parse LLM response as JSON",
                "match_score": 0.0,
                "experience_match": False,
                "skills_match": False,
                "location_match": False,
                "project_experience_match": False,
                "has_github": False,
                "has_tech_publications": False,
                "has_code_contributions": False,
                "key_strengths": [],
                "key_weaknesses": ["Could not analyze resume properly"],
                "rationale": f"Error processing resume: Failed to parse response as JSON. Raw response: {result[:500]}..."
            }
    
    except Exception as e:
        # Return a structured error with traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {
            "error": error_msg,
            "match_score": 0.0,
            "experience_match": False,
            "skills_match": False,
            "location_match": False,
            "project_experience_match": False,
            "has_github": False,
            "has_tech_publications": False,
            "has_code_contributions": False,
            "key_strengths": [],
            "key_weaknesses": ["Could not analyze resume properly"],
            "rationale": f"Error processing resume: {error_msg}. Stack trace: {stack_trace}"
        }

# Function to analyze resume using Gemini API
def analyze_with_gemini(resume_text, requirements, api_key):
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are a professional resume analyzer. Please analyze the following resume against these job requirements:
        
        REQUIREMENTS:
        {requirements}
        
        RESUME:
        {resume_text}
        
        Please provide your analysis in JSON format with the following structure:
        {{
            "match_score": (float between 0 and 1),
            "experience_match": (boolean),
            "skills_match": (boolean),
            "location_match": (boolean),
            "project_experience_match": (boolean),
            "has_github": (boolean),
            "has_tech_publications": (boolean),
            "has_code_contributions": (boolean),
            "key_strengths": (list of strings),
            "key_weaknesses": (list of strings),
            "rationale": (string with detailed explanation)
        }}
        
        Focus on quantifiable metrics and concrete evidence from the resume. Be fair but thorough in your assessment.
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        analysis_text = response.text
        # Find JSON in the response
        match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
        if match:
            analysis_text = match.group(1)
        else:
            # Try to find JSON without code block
            match = re.search(r'({.*})', analysis_text, re.DOTALL)
            if match:
                analysis_text = match.group(1)
            else:
                # Last resort, try to use the entire text
                analysis_text = analysis_text
        
        try:
            return json.loads(analysis_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure with error info
            return {
                "error": "Failed to parse LLM response as JSON",
                "match_score": 0.0,
                "experience_match": False,
                "skills_match": False,
                "location_match": False,
                "project_experience_match": False,
                "has_github": False,
                "has_tech_publications": False,
                "has_code_contributions": False,
                "key_strengths": [],
                "key_weaknesses": ["Could not analyze resume properly"],
                "rationale": f"Error processing resume: Failed to parse response as JSON. Raw response: {analysis_text[:500]}..."
            }
    
    except Exception as e:
        # Return a structured error with traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {
            "error": error_msg,
            "match_score": 0.0,
            "experience_match": False,
            "skills_match": False,
            "location_match": False,
            "project_experience_match": False,
            "has_github": False,
            "has_tech_publications": False,
            "has_code_contributions": False,
            "key_strengths": [],
            "key_weaknesses": ["Could not analyze resume properly"],
            "rationale": f"Error processing resume: {error_msg}. Stack trace: {stack_trace}"
        }
# Main Streamlit Application
def main():
    st.title("AI-Powered Resume Shortlister")
    
    # Create a session state to store the upload_dir
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = Path(tempfile.mkdtemp())
    
    # Cleanup function for when the app is done
    def cleanup_temp_files():
        if hasattr(st.session_state, 'temp_dir') and st.session_state.temp_dir.exists():
            shutil.rmtree(str(st.session_state.temp_dir))
    
    # Register the cleanup function
    import atexit
    atexit.register(cleanup_temp_files)
    
    with st.sidebar:
        st.header("API Configuration")
        llm_choice = st.selectbox(
            "Select LLM Provider",
            ["Claude (Anthropic)", "GPT (OpenAI)", "Gemini (Google)"]
        )
        
        api_key = st.text_input("Enter API Key", type="password")
        
        st.header("Resume Source")
        # File uploader option
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)", 
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            help="Select one or more resume files to analyze"
        )
        
        # Use the session state temp directory to store uploaded files
        if uploaded_files:
            # Clear previous files but don't delete the directory
            for item in st.session_state.temp_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            
            # Save the new uploads
            for uploaded_file in uploaded_files:
                file_path = st.session_state.temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            resume_folder = str(st.session_state.temp_dir)
            
            # Show a message with the number of files uploaded
            st.success(f"Successfully uploaded {len(uploaded_files)} resume file(s)")
        else:
            resume_folder = None
        
        st.header("Job Requirements")
        
        # Job Description Upload Option
        st.subheader("Upload Job Description (Optional)")
        job_description_file = st.file_uploader(
            "Upload Job Description (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"],
            help="Upload a job description file for automatic requirement extraction"
        )
        
        job_description_text = ""
        if job_description_file:
            # Save uploaded JD to temp directory
            jd_file_path = st.session_state.temp_dir / job_description_file.name
            with open(jd_file_path, "wb") as f:
                f.write(job_description_file.getbuffer())
            
            # Extract text from JD file
            job_description_text = extract_text_from_file(str(jd_file_path))
            
            # Show preview of JD
            with st.expander("Preview Job Description"):
                st.write(job_description_text[:1000] + "..." if len(job_description_text) > 1000 else job_description_text)
        
        st.subheader("Manual Requirements (Optional if JD is uploaded)")
        
        # Use a two-column layout for min and max years
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            min_years = st.number_input("Minimum Years of Experience", min_value=0, max_value=25, value=2)
        with exp_col2:
            max_years = st.number_input("Maximum Years of Experience", min_value=min_years, max_value=25, value=min_years + 3)
        
        key_skills = st.text_area("Required Skills (comma separated)")
        location = st.text_input("Preferred Location(s)")
        similar_project_exp = st.text_area("Similar Project Experience")
        
        st.header("Optional Requirements")
        needs_github = st.checkbox("GitHub Profile")
        needs_publications = st.checkbox("Technical Publications")
        needs_code_contributions = st.checkbox("Code/Library Contributions")
        
        match_threshold = st.slider("Match Score Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        
        submit_button = st.button("Shortlist Candidates")
    
    # Convert requirements to structured format
    if job_description_text:
        # If JD is provided, include it at the top of requirements
        requirements_text = f"""
        JOB DESCRIPTION:
        {job_description_text}
        
        Additional Requirements:
        - Years of Experience: {min_years}-{max_years} years
        - Key Skills: {key_skills}
        - Location: {location}
        - Similar Project Experience: {similar_project_exp}
        
        Good to Have (Optional):
        - GitHub Profile: {"Required" if needs_github else "Nice to have"}
        - Technical Publications: {"Required" if needs_publications else "Nice to have"}
        - Code/Library Contributions: {"Required" if needs_code_contributions else "Nice to have"}
        """
    else:
        # If no JD is provided, use manual requirements
        requirements_text = f"""
        Required:
        - Years of Experience: {min_years}-{max_years} years
        - Key Skills: {key_skills}
        - Location: {location}
        - Similar Project Experience: {similar_project_exp}
        
        Good to Have (Optional):
        - GitHub Profile: {"Required" if needs_github else "Nice to have"}
        - Technical Publications: {"Required" if needs_publications else "Nice to have"}
        - Code/Library Contributions: {"Required" if needs_code_contributions else "Nice to have"}
        """
    
    # Main area for results
    if submit_button and api_key:
        if not uploaded_files:
            st.error("Please upload at least one resume file")
            return
        
        resumes = []
        if resume_folder and os.path.isdir(resume_folder):
            for file in os.listdir(resume_folder):
                file_path = os.path.join(resume_folder, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.pdf', '.docx', '.txt')):
                    resumes.append(file_path)
        
        if not resumes:
            st.error("No valid resume files (.pdf, .docx, .txt) found")
            return
        
        st.write(f"Found {len(resumes)} resume files. Beginning analysis...")
        
        # Progress bar
        progress_bar = st.progress(0)
        results = []
        
        # Check for API key and model compatibility
        try:
            if llm_choice == "Claude (Anthropic)":
                # Test connecting to Anthropic API with a minimal call
                client = anthropic.Anthropic(api_key=api_key)
                test_response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": "Test"}
                    ]
                )
                st.success("Successfully connected to Anthropic API")
            elif llm_choice == "GPT (OpenAI)":
                # Test connecting to OpenAI API
                if OPENAI_NEW_VERSION:
                    client = OpenAI(api_key=api_key)
                    models = client.models.list()
                else:
                    openai.api_key = api_key
                    models = openai.Model.list()
                st.success("Successfully connected to OpenAI API")
            elif llm_choice == "Gemini (Google)":
                # Test connecting to Google API
                genai.configure(api_key=api_key)
                # Make a minimal test call
                model = genai.GenerativeModel('gemini-1.5-pro')
                test_response = model.generate_content("Test")
                st.success("Successfully connected to Google Gemini API")
        except Exception as e:
            st.error(f"Error connecting to {llm_choice} API: {str(e)}")
            st.info("Please check your API key and try again.")
            return
            
        st.info(f"Beginning resume analysis...")
        
        for i, resume_path in enumerate(resumes):
            with st.spinner(f"Analyzing {os.path.basename(resume_path)}... ({i+1}/{len(resumes)})"):
                try:
                    resume_text = extract_text_from_file(resume_path)
                    
                    # Show preview of the extracted text
                    with st.expander(f"Preview of {os.path.basename(resume_path)}"):
                        st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
                    
                    # Analyze with selected LLM
                    if llm_choice == "Claude (Anthropic)":
                        analysis = analyze_with_claude(resume_text, requirements_text, api_key)
                    elif llm_choice == "GPT (OpenAI)":
                        analysis = analyze_with_openai(resume_text, requirements_text, api_key)
                    else:  # Gemini
                        analysis = analyze_with_gemini(resume_text, requirements_text, api_key)
                    
                    # Add file name to analysis
                    analysis["file_name"] = os.path.basename(resume_path)
                    
                    # Add to appropriate results list
                    if "error" in analysis:
                        # Add to failed analysis list
                        analysis["status"] = "failed"
                        results.append(analysis)
                        st.error(f"Error analyzing {os.path.basename(resume_path)}: {analysis['error']}")
                    else:
                        # Add to normal results list
                        analysis["status"] = "analyzed"
                        results.append(analysis)
                        st.success(f"Successfully analyzed {os.path.basename(resume_path)}")
                
                except Exception as e:
                    # Catch any unexpected errors and add to failed list
                    error_msg = str(e)
                    stack_trace = traceback.format_exc()
                    failed_analysis = {
                        "file_name": os.path.basename(resume_path),
                        "error": error_msg,
                        "match_score": 0.0,
                        "experience_match": False,
                        "skills_match": False,
                        "location_match": False,
                        "project_experience_match": False,
                        "has_github": False,
                        "has_tech_publications": False,
                        "has_code_contributions": False,
                        "key_strengths": [],
                        "key_weaknesses": ["Failed to process resume"],
                        "rationale": f"Error processing resume: {error_msg}. Stack trace: {stack_trace}",
                        "status": "failed"
                    }
                    results.append(failed_analysis)
                    st.error(f"Error processing {os.path.basename(resume_path)}: {error_msg}")
            
            # Update progress bar
            progress_bar.progress((i + 1) / len(resumes))


# Shortlist candidates
        shortlisted = [r for r in results if r.get("status") == "analyzed" and r.get("match_score", 0) >= match_threshold]
        not_shortlisted = [r for r in results if r.get("status") == "analyzed" and r.get("match_score", 0) < match_threshold]
        failed_analyses = [r for r in results if r.get("status") == "failed"]
        
        # Sort candidates by match score
        shortlisted.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        not_shortlisted.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        # Display results in tabs
        tabs = st.tabs(["Shortlisted Candidates", "Not Shortlisted", "Failed Analyses"])
        
        # Tab 1: Shortlisted candidates
        with tabs[0]:
            st.header("Shortlisted Candidates")
            if shortlisted:
                st.write(f"Found {len(shortlisted)} candidates matching your criteria (threshold: {match_threshold})")
                
                for idx, candidate in enumerate(shortlisted):
                    with st.expander(f"#{idx+1}: {candidate['file_name']} - Match Score: {candidate['match_score']:.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Key Matches")
                            st.write(f"✅ Experience: {'Yes' if candidate.get('experience_match') else 'No'}")
                            st.write(f"✅ Skills: {'Yes' if candidate.get('skills_match') else 'No'}")
                            st.write(f"✅ Location: {'Yes' if candidate.get('location_match') else 'No'}")
                            st.write(f"✅ Project Experience: {'Yes' if candidate.get('project_experience_match') else 'No'}")
                        
                        with col2:
                            st.subheader("Optional Criteria")
                            st.write(f"GitHub: {'Yes' if candidate.get('has_github') else 'No'}")
                            st.write(f"Publications: {'Yes' if candidate.get('has_tech_publications') else 'No'}")
                            st.write(f"Code Contributions: {'Yes' if candidate.get('has_code_contributions') else 'No'}")
                        
                        st.subheader("Strengths")
                        if candidate.get("key_strengths"):
                            for strength in candidate.get("key_strengths"):
                                st.write(f"• {strength}")
                        
                        st.subheader("Areas for Improvement")
                        if candidate.get("key_weaknesses"):
                            for weakness in candidate.get("key_weaknesses"):
                                st.write(f"• {weakness}")
                        
                        st.subheader("Rationale")
                        st.write(candidate.get("rationale", "No rationale provided"))
            else:
                st.warning(f"No candidates met the threshold score of {match_threshold}. Consider lowering the threshold or adjusting requirements.")
        
        # Tab 2: Not shortlisted candidates
        with tabs[1]:
            st.header("Not Shortlisted Candidates")
            if not_shortlisted:
                st.write(f"Found {len(not_shortlisted)} candidates below the threshold score of {match_threshold}")
                
                for idx, candidate in enumerate(not_shortlisted):
                    with st.expander(f"#{idx+1}: {candidate['file_name']} - Match Score: {candidate['match_score']:.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Key Matches")
                            st.write(f"✅ Experience: {'Yes' if candidate.get('experience_match') else 'No'}")
                            st.write(f"✅ Skills: {'Yes' if candidate.get('skills_match') else 'No'}")
                            st.write(f"✅ Location: {'Yes' if candidate.get('location_match') else 'No'}")
                            st.write(f"✅ Project Experience: {'Yes' if candidate.get('project_experience_match') else 'No'}")
                        
                        with col2:
                            st.subheader("Optional Criteria")
                            st.write(f"GitHub: {'Yes' if candidate.get('has_github') else 'No'}")
                            st.write(f"Publications: {'Yes' if candidate.get('has_tech_publications') else 'No'}")
                            st.write(f"Code Contributions: {'Yes' if candidate.get('has_code_contributions') else 'No'}")
                        
                        st.subheader("Strengths")
                        if candidate.get("key_strengths"):
                            for strength in candidate.get("key_strengths"):
                                st.write(f"• {strength}")
                        
                        st.subheader("Areas for Improvement")
                        if candidate.get("key_weaknesses"):
                            for weakness in candidate.get("key_weaknesses"):
                                st.write(f"• {weakness}")
                        
                        st.subheader("Rationale")
                        st.write(candidate.get("rationale", "No rationale provided"))
            else:
                st.info("All analyzed candidates met the threshold score.")
        
        # Tab 3: Failed analyses
        with tabs[2]:
            st.header("Failed Analyses")
            if failed_analyses:
                st.write(f"Failed to analyze {len(failed_analyses)} resume(s)")
                
                for idx, candidate in enumerate(failed_analyses):
                    with st.expander(f"#{idx+1}: {candidate['file_name']} - Failed Analysis"):
                        st.error(f"Error: {candidate.get('error', 'Unknown error')}")
                        st.subheader("Details")
                        st.write(candidate.get("rationale", "No details available"))
            else:
                st.success("All resumes were analyzed successfully.")
        
        # Export results button
        if results:
            # Create export data with separate sections
            export_data = []
            
            # Add shortlisted candidates
            if shortlisted:
                export_data.append(pd.DataFrame([{"Candidate": "SHORTLISTED CANDIDATES", "Match Score": "", "Rationale": ""}]))
                
                shortlisted_data = pd.DataFrame([
                    {
                        "Candidate": candidate["file_name"],
                        "Match Score": candidate["match_score"],
                        "Experience Match": candidate.get("experience_match", False),
                        "Skills Match": candidate.get("skills_match", False),
                        "Location Match": candidate.get("location_match", False),
                        "Project Match": candidate.get("project_experience_match", False),
                        "Has GitHub": candidate.get("has_github", False),
                        "Has Publications": candidate.get("has_tech_publications", False),
                        "Has Code Contributions": candidate.get("has_code_contributions", False),
                        "Strengths": ", ".join(candidate.get("key_strengths", [])),
                        "Weaknesses": ", ".join(candidate.get("key_weaknesses", [])),
                        "Rationale": candidate.get("rationale", "")
                    }
                    for candidate in shortlisted
                ])
                export_data.append(shortlisted_data)
                
                # Add separator
                export_data.append(pd.DataFrame([{"Candidate": "", "Match Score": "", "Rationale": ""}]))
                export_data.append(pd.DataFrame([{"Candidate": "", "Match Score": "", "Rationale": ""}]))
            
            # Add not shortlisted candidates
            if not_shortlisted:
                export_data.append(pd.DataFrame([{"Candidate": "NOT SHORTLISTED CANDIDATES", "Match Score": "", "Rationale": ""}]))
                
                not_shortlisted_data = pd.DataFrame([
                    {
                        "Candidate": candidate["file_name"],
                        "Match Score": candidate["match_score"],
                        "Experience Match": candidate.get("experience_match", False),
                        "Skills Match": candidate.get("skills_match", False),
                        "Location Match": candidate.get("location_match", False),
                        "Project Match": candidate.get("project_experience_match", False),
                        "Has GitHub": candidate.get("has_github", False),
                        "Has Publications": candidate.get("has_tech_publications", False),
                        "Has Code Contributions": candidate.get("has_code_contributions", False),
                        "Strengths": ", ".join(candidate.get("key_strengths", [])),
                        "Weaknesses": ", ".join(candidate.get("key_weaknesses", [])),
                        "Rationale": candidate.get("rationale", "")
                    }
                    for candidate in not_shortlisted
                ])
                export_data.append(not_shortlisted_data)
                
                # Add separator
                export_data.append(pd.DataFrame([{"Candidate": "", "Match Score": "", "Rationale": ""}]))
                export_data.append(pd.DataFrame([{"Candidate": "", "Match Score": "", "Rationale": ""}]))
            
            # Add failed analyses
            if failed_analyses:
                export_data.append(pd.DataFrame([{"Candidate": "FAILED ANALYSES", "Match Score": "", "Rationale": ""}]))
                
                failed_data = pd.DataFrame([
                    {
                        "Candidate": candidate["file_name"],
                        "Error": candidate.get("error", "Unknown error"),
                        "Details": candidate.get("rationale", "No details available")
                    }
                    for candidate in failed_analyses
                ])
                export_data.append(failed_data)
            
            # Combine all data
            combined_csv = pd.concat(export_data, ignore_index=True)
            
            st.download_button(
                label="Export Results as CSV",
                data=combined_csv.to_csv(index=False).encode('utf-8'),
                file_name="resume_analysis_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()