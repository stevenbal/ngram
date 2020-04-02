import collections
import csv

from . import LanguageModel
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, *models):
        """
        Description:    constructor for a Classifier object, given one
                        or more LanguageModel objects

        Input:
        -*models:       tuple, any number of tuples containing the class
                        the LanguageModel object represents, and the
                        LanguageModel object
        """
        # self.models = {model[0]: model[1] for model in models}
        self.models = models

    def __repr__(self):
        """
        Description:    function that shows the representation of the
                        class instance

        Output:
        -object_string: string that shows the model parameters of an
                        instance
        """
        object_string = f"NaiveBayesClassifier({self.models})"
        return object_string

    def classify(self, sentence, mixture=None, prediction_thres=0):
        """
        Description:        function that returns the most probable
                            class for a given sentence

        Input:
        -sentence:          str, the sentence to be classified
        -mixture:           list, contains integers specifying the
                            orders of models to be used as a mixture
                            model, the class is selected by majority
                            vote in this case (default: [])
        -prediction_thres:  float, specifies the minimum required
                            difference divided by the average log
                            probability as produced by the n-gram models
                            to produce a prediction, else the
                            classification returns 'undefined'

        Output:
        -predicted_class:   str, the class as predicted by the
                            classifier
        """
        if mixture:
            if max(mixture) > len(list(self.models)[0].get_models()):
                raise ModelOrderError(max(mixture))
            results = []
            for order in mixture:
                probs = {
                    model.get_class(): model.compute_prob(sentence, N=order)
                    for model in self.models
                }
                sorted_probs = sorted(probs, key=probs.get, reverse=True)
                most_prob, second_most_prob = sorted_probs[0:2]
                diff = abs(probs[most_prob] - probs[second_most_prob])

                mean_abs_log_prob = abs(np.mean(list(probs.values())))
                if diff / mean_abs_log_prob < prediction_thres:
                    predicted_class = "undefined"
                else:
                    predicted_class = most_prob
                results.append(predicted_class)
            class_counts = collections.Counter(results)
            predicted_class = class_counts.most_common(1)[0][0]
            return predicted_class
        else:
            probs = {
                model.get_class(): model.compute_prob(sentence) for model in self.models
            }
            sorted_probs = sorted(probs, key=probs.get, reverse=True)
            most_prob, second_most_prob = sorted_probs[0:2]

            diff = abs(probs[most_prob] - probs[second_most_prob])

            mean_abs_log_prob = abs(np.mean(list(probs.values())))
            if diff / mean_abs_log_prob < prediction_thres:
                predicted_class = "undefined"
            else:
                predicted_class = most_prob
            return predicted_class

    def evaluate(self, filename, mixture=None, prediction_thres=0):
        """
        Description:        function that evaluates the performance of
                            the classifier for a given test corpus

        Input:
        -filename:          str, the name of the csv file containing the
                            test corpus
        -mixture:           list, specifies the mixture model
                            (default: [])
        -prediction_thres:  float, specifies the minimum required
                            difference divided by the average log
                            probability as produced by the n-gram models
                            to produce a prediction, else the
                            classifier returns 'undefined'

        Output:
        -results:           dict, contains the counts of true and false
                            positives and true and false negatives
        """
        results = {
            "positive": {"true": 0, "false": 0},
            "negative": {"true": 0, "false": 0},
        }
        correct_labels = []
        predicted_labels = []
        with open(filename, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            
            # Skip header
            next(reader)
            for row in reader:
                sentiment = row[1]
                correct_labels.append(sentiment)
                prediction = self.classify(
                    row[0], mixture=mixture, prediction_thres=prediction_thres
                )
                predicted_labels.append(prediction)
                if prediction == "undefined":
                    continue
                if prediction == sentiment:
                    results[sentiment]["true"] += 1
                else:
                    results[prediction]["false"] += 1
        precision = self.compute_precision(results)
        recall = self.compute_recall(results)
        print(f"precision {precision}, recall {recall}")
        return precision, recall, results

    def compute_precision(self, results):
        """
        Description:    function that computes precision given the
                        classification results

        Input:
        -results:       dict, contains the counts of true and false
                        positives and true and false negatives

        Output:
        -precision:     float, precision as produced by classifier
        """
        true_pos = results["positive"]["true"]
        false_pos = results["positive"]["false"]
        if true_pos + false_pos:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        return precision

    def compute_recall(self, results):
        """
        Description:    function that computes recall given the
                        classification results

        Input:
        -results:       dict, contains the counts of true and false
                        positives and true and false negatives

        Output:
        -recall:        float, recall as produced by classifier
        """
        true_pos = results["positive"]["true"]
        false_neg = results["negative"]["false"]
        if true_pos + false_neg:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        return recall


class ModelOrderError(ValueError):
    """Indicates that an invaled order for a mixture model was given"""

    pass
