from ngram.nested_dict import NestedDict
from ngram.language_model import LanguageModel
import pytest
import os


class TestLanguageModel:
    @classmethod
    def setup_class(cls):
        path = os.path.join(
            os.getcwd(), "tests/unittest_corpora/processed/positive_training.csv"
        )
        cls.LM_pos = LanguageModel("positive", source=path, stemming=False)
        cls.test_model_path = os.path.join(
            os.getcwd(), "tests/unittest_models/test_model.p"
        )

    def test_init_from_source_corpus(self):
        correct = [
            NestedDict(
                {
                    "<s>": 2,
                    "wow": 1,
                    "this": 1,
                    "is": 1,
                    "great": 2,
                    "</s>": 2,
                    "what": 1,
                    "a": 1,
                    "story": 1,
                }
            ),
            NestedDict(
                {
                    "<s>": {"wow": 1, "what": 1},
                    "wow": {"this": 1},
                    "this": {"is": 1},
                    "is": {"great": 1},
                    "great": {"</s>": 1, "story": 1},
                    "what": {"a": 1},
                    "a": {"great": 1},
                    "story": {"</s>": 1},
                }
            ),
        ]
        for i, model in enumerate(self.LM_pos.get_models()):
            assert model == correct[i]

    def test_save_model_and_init_from_model_file(self):
        self.LM_pos.save_models(self.test_model_path)
        new_model = LanguageModel("positive", model_file=self.test_model_path)

        assert new_model.__dict__ == self.LM_pos.__dict__

    def test_init_N(self):
        pass

    def test_init_stemming(self):
        pass

    def test_init_stopword_removal(self):
        pass

    def test_init_chars(self):
        pass
