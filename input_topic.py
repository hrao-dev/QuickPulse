# input_topic.py


# Input Design
# This script is designed to take user input for a topic or a keyword and validate it before using it in a news summarization application.

def get_topic():
    topic = input("Enter a topic to search for news articles: ")
    if not topic:
        print("No topic provided. Please enter a valid topic.")
        return None
    if len(topic) > 100:  # Arbitrary limit for topic length
        print("Topic is too long. Please enter a shorter topic.")
        return None 
    if not topic.isascii():
        print("Topic contains non-ASCII characters. Please use only ASCII characters.")
        return None
    if not topic.isprintable():
        print("Topic contains non-printable characters. Please use only printable characters.")
        return None
    if topic[0].isdigit():
        print("Topic should not start with a digit. Please enter a valid topic.")
        return None
    if topic[0] == ' ':
        print("Topic should not start with a space. Please enter a valid topic.")
        return None
    # Normalize the input to lowercase and strip any leading/trailing whitespace.
    topic = topic.lower().strip()
    # Check for special characters and replace them with spaces.
    special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '{', '}', '[', ']', '|', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '\\']
    for char in special_chars:
        topic = topic.replace(char, ' ')
    # Remove extra spaces
    topic = ' '.join(topic.split())
    # Check if the topic is empty after normalization
    if not topic:
        print("Topic is empty after normalization. Please enter a valid topic.")
        return None
    # Check for common stop words and remove them
    stop_words = ['the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'on', 'with', 'as', 'by', 'this', 'that']
    topic_words = topic.split()
    topic = ' '.join([word for word in topic_words if word not in stop_words])
    # Check if the topic is empty after removing stop words
    if not topic:
        print("Topic is empty after removing stop words. Please enter a valid topic.")
        return None

    return topic


