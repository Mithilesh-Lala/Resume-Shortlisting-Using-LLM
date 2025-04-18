# Resume-Shortlisting-Using-LLM

A powerful AI-driven application that helps recruiters and hiring managers analyze and shortlist job candidates by evaluating resumes against job requirements using leading LLM providers (Claude, GPT, and Gemini).

## Features

- **Multiple LLM Support**: Choose between Claude (Anthropic), GPT (OpenAI), or Gemini (Google) for resume analysis
- **Job Description Upload**: Upload a job description to automatically extract requirements
- **Flexible Resume Formats**: Supports PDF, DOCX, and TXT resume formats
- **Simple File Upload**: Upload multiple resume files directly through the browser
- **Experience Range Selection**: Specify a range for years of experience (e.g., 2-5 years)
- **Customizable Requirements**: Specify skills, location, and project requirements
- **Optional Criteria**: Filter for GitHub profiles, technical publications, and code contributions
- **Comprehensive Results**: View shortlisted, not shortlisted, and failed analyses in separate tabs
- **Detailed Analysis**: Get comprehensive rationales for each candidate
- **Robust Error Handling**: Application continues even if some resume analyses fail
- **Export Results**: Download all results as a CSV file with separate sections




## Installation

### Prerequisites

1. Python 3.8 or higher
2. API key from at least one of the supported LLM providers:
   - Anthropic (Claude)
   - OpenAI (GPT)
   - Google (Gemini)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/Mithilesh-Lala/Resume-Shortlisting-Using-LLM.git
cd Resume-Shortlisting-Using-LLM
```

2. Install the required packages:
```bash
pip install streamlit pandas anthropic google-generativeai openai PyPDF2 python-docx docx2txt
```

Note: The application supports both newer (≥1.0.0) and older versions of the OpenAI Python package.

### Running the Application

Run the Streamlit application:
```bash
streamlit run main.py
```

## How to Use

1. **Configure API**: Select your preferred LLM provider and enter your API key
2. **Upload Resumes**: Directly upload PDF, DOCX, or TXT files through the interface
3. **Upload Job Description (Optional)**: Upload a job description file to automatically extract requirements
4. **Specify Requirements**:
   - Set experience range with minimum and maximum years
   - List required skills (comma separated)
   - Enter preferred location(s)
   - Describe similar project experience the candidates should have
5. **Set Optional Requirements**:
   - Check if GitHub profile is required
   - Check if technical publications are required
   - Check if code/library contributions are required
6. **Adjust Match Threshold**: Set the minimum match score (0.0-1.0) for shortlisting
7. **Run Analysis**: Click "Shortlist Candidates" to begin processing
8. **Review Results**: Examine candidates across three tabs:
   - Shortlisted Candidates: Those who met or exceeded the threshold score
   - Not Shortlisted: Those who were successfully analyzed but scored below the threshold
   - Failed Analyses: Resumes that could not be analyzed properly
9. **Export Data**: Download comprehensive results as a CSV file with all three categories

## Tips for Best Results

1. **Job Description First**: Upload a detailed job description for the most accurate matching
2. **Experience Range**: Set a realistic experience range that matches your actual requirements
3. **Be Specific with Skills**: List specific technical skills rather than general terms
4. **Try Different LLMs**: Each LLM may have different strengths in analyzing resumes
5. **Adjust Threshold**: Start with a higher threshold (0.7-0.8) and lower if needed
6. **Review Rationales**: Always check the LLM's reasoning to understand the match scores

## LLM Provider Information

### Claude (Anthropic)
- Uses Claude 3.5 Sonnet model
- Best for nuanced understanding of resume content
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

### GPT (OpenAI)
- Uses GPT-4 Turbo model
- Strong general performance across various resume formats
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

### Gemini (Google)
- Uses Gemini 1.5 Pro model
- Good at extracting structured information
- [Google AI Studio Documentation](https://ai.google.dev/docs)

## Security Considerations

- API keys are never stored persistently by the application
- Resume files are temporarily stored during the session and deleted afterward
- All processing happens on your local machine

## CSV Export Format

The CSV export contains three sections:
1. **SHORTLISTED CANDIDATES**: Those meeting or exceeding the threshold
2. **NOT SHORTLISTED CANDIDATES**: Those falling below the threshold
3. **FAILED ANALYSES**: Resumes that couldn't be analyzed with error details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit, a powerful framework for data applications
- Utilizes advanced LLM APIs from Anthropic, OpenAI, and Google
- Special thanks to the open-source libraries that made this possible
