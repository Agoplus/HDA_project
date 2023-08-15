import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_fscore_support, RocCurveDisplay, roc_auc_score
import tensorflow as tf
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import optuna
import pickle


def model_evaluation(train_predictions, val_predictions, test_predictions,
                     train_labels, val_labels, test_labels, num_patches_per_image):
    """
    Computes the evaluation metrics for the model, plots some graphs and saves them to the disk.
    :param train_predictions: numpy array of shape (num_imgs*num_patches_per_image)
    :param val_predictions: numpy array of shape (num_imgs*num_patches_per_image)
    :param test_predictions: numpy array of shape (num_imgs*num_patches_per_image)
    :param train_labels: numpy array of shape (num_imgs)
    :param val_labels: numpy array of shape (num_imgs)
    :param test_labels: numpy array of shape (num_imgs)
    :param num_patches_per_image: int - number of patches per image
    :return:
    """

    # compute softmax
    train_predictions_proba = np.exp(train_predictions) / np.sum(np.exp(train_predictions), axis=1,
                                                                 keepdims=True)
    val_predictions_proba = np.exp(val_predictions) / np.sum(np.exp(val_predictions), axis=1,
                                                             keepdims=True)
    test_predictions_proba = np.exp(test_predictions) / np.sum(np.exp(test_predictions), axis=1,
                                                               keepdims=True)

    # compute the mean probability for each image (for roc curve)
    train_predictions_proba_img = train_predictions_proba.reshape(-1, num_patches_per_image, 3)
    train_predictions_proba_img = np.mean(train_predictions_proba_img, axis=1)
    val_predictions_proba_img = val_predictions_proba.reshape(-1, num_patches_per_image, 3)
    val_predictions_proba_img = np.mean(val_predictions_proba_img, axis=1)
    test_predictions_proba_img = test_predictions_proba.reshape(-1, num_patches_per_image, 3)
    test_predictions_proba_img = np.mean(test_predictions_proba_img, axis=1)

    # binarize labels
    train_labels_onehot = LabelBinarizer().fit_transform(train_labels)
    val_labels_onehot = LabelBinarizer().fit_transform(val_labels)
    test_labels_onehot = LabelBinarizer().fit_transform(test_labels)

    # get the class with the highest probability
    train_predictions = np.argmax(train_predictions, axis=1)
    val_predictions = np.argmax(val_predictions, axis=1)
    test_predictions = np.argmax(test_predictions, axis=1)

    # majority voting
    train_predictions = train_predictions.reshape(-1, num_patches_per_image)
    val_predictions = val_predictions.reshape(-1, num_patches_per_image)
    test_predictions = test_predictions.reshape(-1, num_patches_per_image)

    train_predictions = stats.mode(train_predictions, axis=1)[0].reshape(-1)
    val_predictions = stats.mode(val_predictions, axis=1)[0].reshape(-1)
    test_predictions = stats.mode(test_predictions, axis=1)[0].reshape(-1)

    # compute accuracy
    train_acc = accuracy_score(train_labels, train_predictions)
    val_acc = accuracy_score(val_labels, val_predictions)
    test_acc = accuracy_score(test_labels, test_predictions)

    print('Train accuracy: {:.2f}%'.format(train_acc * 100))
    print('Val accuracy: {:.2f}%'.format(val_acc * 100))
    print('Test accuracy: {:.2f}%'.format(test_acc * 100))

    # compute confusion matrix
    train_cm = confusion_matrix(train_labels, train_predictions)
    val_cm = confusion_matrix(val_labels, val_predictions)
    test_cm = confusion_matrix(test_labels, test_predictions)

    cm_display = ConfusionMatrixDisplay(train_cm, display_labels=['CLL', 'FL', 'MCL']).plot(
        cmap='Blues')
    cm_display = ConfusionMatrixDisplay(val_cm, display_labels=['CLL', 'FL', 'MCL']).plot(
        cmap='Blues')
    cm_display = ConfusionMatrixDisplay(test_cm, display_labels=['CLL', 'FL', 'MCL']).plot(
        cmap='Blues')

    # compute precision, recall, f1-score
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels,
                                                                                 train_predictions,
                                                                                 average='macro')
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels,
                                                                           val_predictions,
                                                                           average='macro')
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels,
                                                                              test_predictions,
                                                                              average='macro')

    print('Train precision: {:.2f}%'.format(train_precision * 100),
          'Train recall: {:.2f}%'.format(train_recall * 100),
          'Train f1-score: {:.2f}%'.format(train_f1 * 100))
    print('Val precision: {:.2f}%'.format(val_precision * 100),
          'Val recall: {:.2f}%'.format(val_recall * 100),
          'Val f1-score: {:.2f}%'.format(val_f1 * 100))
    print('Test precision: {:.2f}%'.format(test_precision * 100),
          'Test recall: {:.2f}%'.format(test_recall * 100),
          'Test f1-score: {:.2f}%'.format(test_f1 * 100))

    # compute roc auc score
    train_roc_auc = roc_auc_score(train_labels_onehot, train_predictions_proba_img, average='macro',
                                  multi_class='ovr')
    val_roc_auc = roc_auc_score(val_labels_onehot, val_predictions_proba_img, average='macro',
                                multi_class='ovr')
    test_roc_auc = roc_auc_score(test_labels_onehot, test_predictions_proba_img, average='macro',
                                 multi_class='ovr')

    print('Train ROC AUC: {:.2f}%'.format(train_roc_auc * 100))
    print('Val ROC AUC: {:.2f}%'.format(val_roc_auc * 100))
    print('Test ROC AUC: {:.2f}%'.format(test_roc_auc * 100))

    class_names = ['CLL', 'FL', 'MCL']

    for class_id in range(3):
        RocCurveDisplay.from_predictions(
            test_labels_onehot[:, class_id],
            test_predictions_proba_img[:, class_id],
            name=f"{class_names[class_id]} vs the rest",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{class_names[class_id]}-vs-the rest ROC curves")
        plt.legend()
        plt.show()


class ModelEvaluation:

    def __init__(self, dataset, model, classifier_type, seed, classifier_params=None):
        """
        :param dataset: Dataset object
        :param model: tf model for feature extraction, with loaded weights
        :param classifier_type: str - 'xgboost' or 'svm'
        :param seed: int - random seed
        :param classifier_params: dict - parameters for the classifier if already optimized
        """
        self.dataset = dataset
        self.model = model
        assert classifier_type in ['xgboost', 'svm'], 'classifier_type must be xgboost or svm'
        self.classifier_type = classifier_type
        self.seed = seed
        self.classifier_params = classifier_params

        self.num_patches_per_image = dataset.num_patches_per_image
        self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_datasets()
        self.train_steps, self.val_steps, self.test_steps = dataset.get_steps()

        self.train_labels, self.val_labels, self.test_labels, self.train_img_labels, \
            self.val_img_labels, self.test_img_labels = self._get_labels()

        self.train_features, self.val_features, self.test_features = None, None, None
        self.classifier = None

    def _get_labels(self):
        train_labels = []
        for _, labels in self.train_dataset.take(self.train_steps + 1):
            train_labels.append(labels.numpy())
        train_labels = np.concatenate(train_labels)
        train_labels = train_labels[:len(self.dataset.train_split) * self.num_patches_per_image]
        train_img_labels = train_labels[::self.num_patches_per_image]

        val_labels = []
        for _, labels in self.val_dataset.take(self.val_steps + 1):
            val_labels.append(labels.numpy())
        val_labels = np.concatenate(val_labels)
        val_labels = val_labels[:len(self.dataset.val_split) * self.num_patches_per_image]
        val_img_labels = val_labels[::self.num_patches_per_image]

        test_labels = []
        for _, labels in self.test_dataset.take(self.test_steps + 1):
            test_labels.append(labels.numpy())
        test_labels = np.concatenate(test_labels)
        test_labels = test_labels[:len(self.dataset.test_split) * self.num_patches_per_image]
        test_img_labels = test_labels[::self.num_patches_per_image]

        return train_labels, val_labels, test_labels, train_img_labels, val_img_labels, test_img_labels

    def extract_features(self, output_index):
        """
        Extract features from model for train, val and test datasets
        :param output_index: negative int - index of output layer to extract features from
        :return:
        """
        output = self.model.layers[output_index].output
        feature_extractor = tf.keras.Model(inputs=self.model.input, outputs=output)

        train_features = feature_extractor.predict(self.train_dataset, steps=self.train_steps + 1)
        val_features = feature_extractor.predict(self.val_dataset, steps=self.val_steps + 1)
        test_features = feature_extractor.predict(self.test_dataset, steps=self.test_steps + 1)

        train_features = train_features[:len(self.dataset.train_split) * self.num_patches_per_image]
        val_features = val_features[:len(self.dataset.val_split) * self.num_patches_per_image]
        test_features = test_features[:len(self.dataset.test_split) * self.num_patches_per_image]

        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features

    def _svm_objective(self, trial):
        """
        Objective function for SVM hyperparameter optimization
        :param trial: optuna trial
        :return: float - validation accuracy
        """

        C = trial.suggest_float('C', 1e-6, 1e6, log=True)
        gamma = trial.suggest_float('gamma', 1e-6, 1e6, log=True)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid', 'poly', 'linear'])
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 1, 5)
        else:
            degree = 3

        svm = SVC(C=C,
                  gamma=gamma,
                  kernel=kernel,
                  degree=degree,
                  random_state=self.seed,
                  break_ties=True,
                  decision_function_shape='ovr',
                  probability=True)

        svm.fit(self.train_features, self.train_labels)
        accuracy = svm.score(self.val_features, self.val_labels)

        return accuracy

    def _xgboost_objective(self, trial):
        """
        Objective function for XGBoost hyperparameter optimization
        :param trial: optuna trial
        :return: float - validation accuracy
        """

        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        eta = trial.suggest_float('eta', 1e-6, 0.3, log=True)
        gamma = trial.suggest_float('gamma', 0, 20)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        min_child_weight = trial.suggest_float('min_child_weight', 0, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 100)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 100)

        xgb = XGBClassifier(n_estimators=n_estimators,
                            eta=eta,
                            gamma=gamma,
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha,
                            n_jobs=-1,
                            verbosity=1,
                            objective='multi:softprob',
                            tree_method='gpu_hist',
                            seed=self.seed)

        xgb.fit(self.train_features, self.train_labels)
        accuracy = xgb.score(self.val_features, self.val_labels)

        return accuracy

    def _optimize_classifier_params(self):

        if self.classifier_type == 'svm':
            study = optuna.create_study(direction='maximize')
            study.optimize(self._svm_objective, n_trials=100, n_jobs=-1)
            best_params = study.best_params
            if 'degree' not in best_params:
                best_params['degree'] = 3

            self.classifier = SVC(**best_params,
                                  random_state=self.seed,
                                  break_ties=True,
                                  decision_function_shape='ovr',
                                  probability=True)
            print('Best svm params: ', best_params)
            pickle.dump(self.classifier.get_params(), open('svm_best_params.pkl', 'wb'))

        elif self.classifier_type == 'xgboost':
            study = optuna.create_study(direction='maximize')
            study.optimize(self._xgboost_objective, n_trials=100, n_jobs=1)
            best_params = study.best_params
            self.classifier = XGBClassifier(**best_params,
                                            n_jobs=-1,
                                            verbosity=1,
                                            objective='multi:softprob',
                                            tree_method='gpu_hist',
                                            seed=self.seed)
            print('Best xgboost params: ', best_params)
            pickle.dump(self.classifier.get_params(), open('xgboost_best_params.pkl', 'wb'))

    def train_classifier(self, optimize_params=True):
        """
        Train classifier on extracted features
        :param optimize_params: bool - whether to optimize classifier params using optuna
        :return:
        """

        # check if features have been extracted
        if self.train_features is None:
            print('features have not been extracted, run extract_features() first')
            return None

        if not optimize_params:  # use default params or provided params
            if self.classifier_type == 'xgboost':
                if self.classifier_params is not None:
                    self.classifier = XGBClassifier(**self.classifier_params)
                else:
                    self.classifier = XGBClassifier(n_estimators=100,
                                                    max_depth=3,
                                                    learning_rate=0.1,
                                                    subsample=0.8,
                                                    colsample_bytree=0.8,
                                                    random_state=self.seed,
                                                    n_jobs=-1,
                                                    verbosity=1,
                                                    objective='multi:softmax',
                                                    tree_method='gpu_hist')
            else:
                if self.classifier_params is not None:
                    self.classifier = SVC(**self.classifier_params)
                else:
                    self.classifier = SVC(C=1,
                                          kernel='rbf',
                                          gamma='scale',
                                          break_ties=True,
                                          random_state=self.seed,
                                          decision_function_shape='ovr',
                                          probability=True)
        else:
            self._optimize_classifier_params()

        self.classifier.fit(self.train_features, self.train_labels)
        pickle.dump(self.classifier, open('classifier.pkl', 'wb'))

    def test_classifier(self):
        """
        Obtain classifier performance
        """

        if self.classifier is None:
            print('classifier has not been trained, runnning train_classifier() now')
            self.train_classifier()

        # get predictions
        train_preds = self.classifier.predict_proba(self.train_features)
        val_preds = self.classifier.predict_proba(self.val_features)
        test_preds = self.classifier.predict_proba(self.test_features)

        # run metrics computations
        model_evaluation(train_preds, val_preds, test_preds, self.train_img_labels,
                         self.val_img_labels, self.test_img_labels, self.num_patches_per_image)

    def test_feature_extractor(self):
        """
        Test feature extractor by itself, using the linear layer
        """

        # get predictions
        train_preds = self.model.predict(self.train_dataset, steps=self.train_steps + 1)
        val_preds = self.model.predict(self.val_dataset, steps=self.val_steps + 1)
        test_preds = self.model.predict(self.test_dataset, steps=self.test_steps + 1)

        train_preds = train_preds[:len(self.dataset.train_split) * self.num_patches_per_image]
        val_preds = val_preds[:len(self.dataset.val_split) * self.num_patches_per_image]
        test_preds = test_preds[:len(self.dataset.test_split) * self.num_patches_per_image]

        # run metrics computations
        model_evaluation(train_preds, val_preds, test_preds, self.train_img_labels,
                         self.val_img_labels, self.test_img_labels, self.num_patches_per_image)
