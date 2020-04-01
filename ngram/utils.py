import re


def preprocess_sentence(sentence):
    """
    Description:    function that preprocesses a sentence before it
                    is used in a LanguageModel or classified

    Input:
    -sentence:      str, the sentence to be preprocessed

    Output:
    -sentence:      str, the sentence after preprocessing

    """
    sentence = sentence.lower()
    # Remove website links
    sentence = re.sub(r"(https?:\/\/(www)?[^\s]*)|(www\.[^\s]*)", "", sentence)
    # Remove twitter mentions and email addresses
    sentence = re.sub(r"[^\s]*@[^\s]*", "", sentence)
    # Break up slashed and hyphenated words
    sentence = re.sub(r"(?<=[^\s])(\/|-)(?=[^\s])", " ", sentence)
    # Remove all non alphanumeric characters
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    # Split and join with a single space between words
    sentence = " ".join(sentence.split())
    return sentence
