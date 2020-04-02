import re


def preprocess_sentence(
    sentence,
    lower=True,
    remove_links=True,
    remove_emails=True,
    break_words=True,
    alphanumeric_only=True,
):
    """
    Description:    function that preprocesses a sentence before it
                    is used in a LanguageModel or classified

    Input:
    -sentence:      str, the sentence to be preprocessed

    Output:
    -sentence:      str, the sentence after preprocessing

    """
    if lower:
        sentence = sentence.lower()
    if remove_links:
        # Remove website links
        sentence = re.sub(r"(https?:\/\/(www)?[^\s]*)|(www\.[^\s]*)", "", sentence)
    if remove_emails:
        # Remove twitter mentions and email addresses
        sentence = re.sub(r"[^\s]*@[^\s]*", "", sentence)
    if break_words:
        # Break up slashed and hyphenated words
        sentence = re.sub(r"(?<=[^\s])(\/|-)(?=[^\s])", " ", sentence)
    if alphanumeric_only:
        # Remove all non alphanumeric characters
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        
    # Split and join with a single space between words
    sentence = " ".join(sentence.split())
    return sentence
