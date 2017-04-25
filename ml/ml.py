import joblib
import os
import pandas as pd
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import naive_bayes


def prepare():
    location = os.path.expanduser("~/git/fund/data/general/all_averages.xlsx")
    columns = ["week",
               "1w_avg",
               "2w_avg",
               "1m_avg",
               "2m_avg",
               "1w_dif",
               "2w_dif",
               "1m_dif",
               "2m_dif",
               "pxch1",
               "pxch5",
               "pxch10",
               "pxch5_1",
               "pxch10_1",
               "company",
               "econ_avg",
               "env_avg",
               "lead_avg",
               "legal_avg",
               "soc_avg",
               "source_avg",
               "sign"]
    all = pd.read_excel(location, names=columns)
    return all


def reg_bayesian_1w_avg(df):
    classifier = linear_model.BayesianRidge()
    feat = ["1w_avg"]
    X = df[feat]
    y = df["pxch1"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def reg_bayesian_themes(df):
    classifier = linear_model.BayesianRidge()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg"]
    X = df[feat]
    y = df["pxch1"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def reg_bayesian_themes_best(df):
    classifier = linear_model.BayesianRidge()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg"]
    X = df[feat]
    y = df["pxch1"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X, y)
    selected = get_selected_features(fs.get_support(), feat)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def reg_bayesian_source(df):
    classifier = linear_model.BayesianRidge()
    feat = ["source_avg"]
    X = df[feat]
    y = df["pxch1"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_1w_avg(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["1w_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_themes_best(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg"]
    X = df[feat]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X, y)
    selected = get_selected_features(fs.get_support(), feat)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_themes(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_econ(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_env(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["env_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_lead(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["lead_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_legal(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["legal_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_soc(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["soc_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_source(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["source_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_themes_source(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg", "source_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_themes_source_best(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg", "source_avg"]
    X = df[feat]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X, y)
    selected = get_selected_features(fs.get_support(), feat)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_1w_avg_themes_source_best(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["1w_avg", "econ_avg", "env_avg", "lead_avg", "legal_avg", "soc_avg", "source_avg"]
    X = df[feat]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=5)
    X_new = fs.fit_transform(X, y)
    selected = get_selected_features(fs.get_support(), feat)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_econ_lead_source(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["econ_avg", "lead_avg", "source_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def class_bayesian_sans_env(df):
    classifier = naive_bayes.GaussianNB()
    feat = ["1w_avg", "econ_avg", "lead_avg", "legal_avg", "soc_avg"]
    X = df[feat]
    y = df["sign"]
    fs = None
    selected = None

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y, test_size=0.3, random_state=200)
    classifier.fit(X_train, y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return classifier, feat, train_acc, test_acc, selected if fs else None


def get_selected_features(ndarr, features):
    return [feat if bool else "NULL" for bool, feat in zip(ndarr.tolist(), features)]


def main():
    df = prepare()
    res = []
    res.append(reg_bayesian_1w_avg(df))
    res.append(reg_bayesian_themes(df))
    res.append(reg_bayesian_themes_best(df))
    res.append(class_bayesian_1w_avg(df))
    res.append(class_bayesian_themes(df))
    res.append(class_bayesian_themes_best(df))
    res.append(class_bayesian_econ(df))
    res.append(class_bayesian_env(df))
    res.append(class_bayesian_lead(df))
    res.append(class_bayesian_legal(df))
    res.append(class_bayesian_soc(df))
    res.append(class_bayesian_source(df))
    res.append(class_bayesian_themes_source(df))
    res.append(class_bayesian_themes_source_best(df))  # econ, env, lead, source
    res.append(class_bayesian_1w_avg_themes_source_best(df))
    res.append(class_bayesian_econ_lead_source(df))
    res.append(class_bayesian_sans_env(df))

    list.sort(res, reverse=True, key=lambda tup: tup[3].split(" ")[2])
    for r in res:
        print(r)

    joblib.dump(res[0][1][2], "best_model.pkl")


if __name__ == "__main__":
    main()
