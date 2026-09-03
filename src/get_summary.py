def generate_detailed_summary(input_text, llm_model):
    """
    Generate a detailed, well-explained summary of the input text using a Groq LLM.

    """

    
    prompt = f"""
You are an expert research analyst and technical writer.
Your task is to carefully read the following text and generate a comprehensive, structured summary that covers all key ideas, concepts, and insights. No preamble.

Instructions:

1. Provide a structured summary including:
   - Main idea or theme of the text
   - Important subtopics or sections
   - Key findings, facts, or arguments
   - Any examples or data mentioned

2. Explain complex terms or concepts in a simple and intuitive way, as if teaching someone new to the topic.

3. Ensure clarity and depth — avoid vague or generic summaries.

4. Present the output in a clear format with headings, bullet points, and short paragraphs.

5. If the text is technical or academic, include a section: "Explanation in Simple Terms".

Input Text:
{input_text}

Output Format:
- Title or Theme
- Summary (well-structured paragraphs)
- Key Points (bullet format)
- Explanation in Simple Terms (for layperson understanding)
    """

    response = llm_model.invoke(prompt)

    if isinstance(response, str):
        return response.strip()
    elif hasattr(response, "content"):
        return response.content.strip()
    elif hasattr(response, "text"):
        return response.text.strip()
    else:
        return str(response).strip()