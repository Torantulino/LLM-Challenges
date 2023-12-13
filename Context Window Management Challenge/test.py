from llm import llm

questions = [
    "What is a context window in the context of a language learning model?",
    "How does the size of the context window affect the performance of an LLM?",
    "Can you provide a basic Python example of accessing the context window in an LLM?",
    "What are common techniques for summarizing the content within a context window?",
    "How can I implement a summary algorithm for a context window in Python?",
    "What libraries or tools in Python are best suited for summarizing text data?",
    "Are there any pre-built models or frameworks in Python for context window summarization?",
    "How can the quality of a summary be evaluated in terms of its effectiveness for an LLM?",
    "Can you suggest ways to optimize the summarization process to handle larger context windows?",
    "What are the best practices for integrating context window summaries back into LLM interactions?",
]

system_prompts = [
    "Explain the trade-offs between accuracy and computational efficiency in summarizing context windows for LLMs.",
    "Demonstrate how to use Python's regular expressions to preprocess text data before summarizing for an LLM's context window.",
    "Discuss the impact of different natural language processing techniques on the quality of context window summarization.",
    "Provide a Python code snippet showing how to implement a sliding window technique for dynamic context window management in LLMs.",
    "Explain how to use Python's NLTK library for tokenizing and summarizing text in an LLM's context window.",
    "Illustrate the use of deep learning models in Python for generating context window summaries, comparing RNNs and Transformers.",
    "Describe how to integrate external APIs or services for enhanced context window summarization in Python-based LLMs.",
    "Show how to apply sentiment analysis in Python to the content of a context window to influence the summarization process.",
    "Explain the role of context window summarization in multilingual LLMs and how to handle language-specific nuances in Python.",
    "Demonstrate how to use Python's Pandas library for data analysis to optimize the summarization process based on historical LLM interactions.",
]


def test_process_question():
    chat_helper = llm()

    # We'll test this using several questions mixed with system prompts
    # to see how the model responds and to make sure the system prompts
    # getting summarized and are not too long.

    for question, system_prompt in zip(questions, system_prompts):
        print("\n\033[92mYou:\033[0m", question)
        response = chat_helper.send_message(question)
        print("\n\033[95mAI:", response, "\033[0m")

        print("\n\033[92mSystem:\033[0m", system_prompt)
        chat_helper.send_message(system_prompt, role="system")
        print("\n\033[95mAI:", response, "\033[0m")

if __name__ == "__main__":
    test_process_question()
