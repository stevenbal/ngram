from ngram.utils import preprocess_sentence
import pytest


class TestUtils:
    def test_preprocess_urls(self):
        sentence = """
        http://www.teavcx123ald.nl/test
        https://www.2ews12en.nl.co/als
        www.13lls2.nl
        http://sksmwl244.nl
        www.pasd2p23en.nl
        """
        result = preprocess_sentence(sentence)

        assert result == ""

    def test_preprocess_email_and_twitter_mentions(self):
        sentence = """
        tes @userna2123me
        t23est@sit32e.nl
        """
        result = preprocess_sentence(sentence)

        assert result == "tes"

    def test_preprocess_slashed_and_hyphenated(self):
        sentence = """
        him/her
        this/that
        sea-lions
        """
        result = preprocess_sentence(sentence)

        assert result == "him her this that sea lions"

    def test_preprocess_nonalphanumeric_removal(self):
        sentence = """
        willbe!#$%^&*()_+ removed
        multiple    spaces   test
        gooed&*aal
        """
        result = preprocess_sentence(sentence)

        assert result == "willbe removed multiple spaces test gooedaal"
