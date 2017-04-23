import joblib
import os
import pandas as pd
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import naive_bayes
import numpy

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
    all = pd.read_excel(location,names=columns)
    return all

# def linear_1w_avg_pxch1(df): # 1w_avg -> pxch1
#     classifier = linear_model.LinearRegression()
#
#     X = numpy.expand_dims(df["1w_avg"].values, axis=1)
#     y = numpy.expand_dims(df["pxch1"].values, axis=1)
#
#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
#                                                                          y, test_size=0.3, random_state=200)
#     classifier.fit(X_train,y_train)
#
#     train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
#     test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
#     return classifier,train_acc,test_acc

def reg_bayesian_1w_avg(df):
    classifier = linear_model.BayesianRidge()

    X = numpy.expand_dims(df["1w_avg"].values, axis=1)
    y = df["pxch1"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def reg_bayesian_themes(df):
    classifier = linear_model.BayesianRidge()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg"]]
    y = df["pxch1"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def reg_bayesian_themes_best(df):
    classifier = linear_model.BayesianRidge()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg"]]
    y = df["pxch1"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X, y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def reg_bayesian_source(df):
    classifier = linear_model.BayesianRidge()

    X = df[["source_avg"]]
    y = df["pxch1"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_1w_avg(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["1w_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_themes_best(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg"]]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X,y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_themes(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_econ(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_env(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["env_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_lead(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["lead_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_legal(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["legal_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_soc(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["soc_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_source(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["source_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_themes_source(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg","source_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_themes_source_best(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg","env_avg","lead_avg","legal_avg","soc_avg","source_avg"]]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=4)
    X_new = fs.fit_transform(X, y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc,classifier

def class_bayesian_1w_avg_themes_source_best(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["1w_avg","econ_avg","env_avg","lead_avg","legal_avg","soc_avg","source_avg"]]
    y = df["sign"]
    fs = feature_selection.SelectKBest(k=5)
    X_new = fs.fit_transform(X, y)
    # print(fs.get_support())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_econ_lead_source(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["econ_avg","lead_avg","source_avg"]]
    y = df["sign"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def class_bayesian_sans_env(df):
    classifier = naive_bayes.GaussianNB()

    X = df[["1w_avg","econ_avg","lead_avg","legal_avg","soc_avg"]]
    y = df["sign"]
    # fs = feature_selection.SelectKBest(k=3)
    # X_new = fs.fit_transform(X,y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                         y, test_size=0.3, random_state=200)
    classifier.fit(X_train,y_train)

    train_acc = 'train accuracy: {:.4f}'.format(classifier.score(X_train, y_train))
    test_acc = 'test accuracy: {:.4f}'.format(classifier.score(X_test, y_test))
    return train_acc,test_acc

def main():
    df = prepare()
    res = []
    res.append(("reg 1w_avg",reg_bayesian_1w_avg(df)))
    res.append(("reg themes",reg_bayesian_themes(df)))
    res.append(("reg best themes",reg_bayesian_themes_best(df)))
    res.append(("class 1w_avg",class_bayesian_1w_avg(df)))
    res.append(("class themes",class_bayesian_themes(df)))
    res.append(("class best themes",class_bayesian_themes_best(df)))
    res.append(("class econ",class_bayesian_econ(df)))
    res.append(("class env",class_bayesian_env(df)))
    res.append(("class lead",class_bayesian_lead(df)))
    res.append(("class legal",class_bayesian_legal(df)))
    res.append(("class soc",class_bayesian_soc(df)))
    res.append(("class source",class_bayesian_source(df)))
    res.append(("class themes+sources",class_bayesian_themes_source(df)))
    res.append(("class best themes+sources",class_bayesian_themes_source_best(df))) # econ, env, lead, source
    res.append(("class best 1w_avg+themes+sources",class_bayesian_1w_avg_themes_source_best(df)))
    res.append(("class econ+lead+source",class_bayesian_econ_lead_source(df)))
    res.append(("class sans env",class_bayesian_sans_env(df)))

    list.sort(res,reverse=True,key =lambda tup: tup[1][1].split(" ")[2])
    for r in res:
        print(r)
    print(len(res))

    joblib.dump(res[0][1][2],"best_model.pkl")
if __name__ == "__main__":
    main()