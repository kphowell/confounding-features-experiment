# -*- coding: utf-8 -*-
"""
Created on July 12, 2019

Anonymous script for AAAI Health Intelligence 2020 Workshop Review
"""

import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy
from scipy import sparse
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
import copy
from matplotlib import pyplot
from collections import OrderedDict
import random
from sklearn.feature_selection import chi2
from scipy import stats

NOTE_TEXT_INDEX = 1
GOC_INDEX = 2
INPATIENT_INDEX = 3
DATASET_INDEX = 4
NOTE_TYPE_INDEX = 7
FEATURE_INDEX = 8


class BackdoorAdjustment:
    """This class is adapted from github.com/tapilab/aaai-2016-robust"""
    def __init__(self):
        self.clf = LogisticRegression(class_weight='balanced')

    def predict_proba(self, X):
        # build features with every possible confounder
        l = X.shape[0]
        rows = range(l * self.count_c)
        cols = list(range(self.count_c)) * l
        data = [self.c_ft_value] * (l * self.count_c)
        c = sparse.csr_matrix((data, (rows, cols)))
        # build the probabilities to be multiplied by
        p = numpy.array(self.c_prob).reshape(-1, 1)
        p =numpy.tile(p, (X.shape[0], 1))

        # combine the original features and the possible confounder values
        repeat_indices = numpy.arange(X.shape[0]).repeat(self.count_c)
        X = X[repeat_indices]
        Xc = sparse.hstack((X, c), format='csr')
        proba = self.clf.predict_proba(Xc)
        # multiply by P(z) and sum over the confounder for every instance in X
        proba *= p
        proba = proba.reshape(-1, self.count_c, self.count_y)
        proba = numpy.sum(proba, axis=1)
        # normalize
        norm = numpy.sum(proba, axis=1).reshape(-1, 1)
        proba /= norm
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        prediction = numpy.array(proba.argmax(axis=1))
        return prediction[0], proba[0][1]


    def fit(self, X, y, c, c_ft_value=1.):
        self.c_prob = numpy.bincount(c) / len(c)
        self.c_ft_value = c_ft_value
        self.count_c = len(set(c))
        self.count_y = len(set(y))

        rows = range(len(c))
        cols = c
        data = [c_ft_value] * len(c)
        c_fts = sparse.csr_matrix((data, (rows, cols)))
        Xc = sparse.hstack((X, c_fts), format='csr')

        self.clf.fit(Xc, y)


def backdoor_adjustment_var_C(X, y, z, c):
    clf = BackdoorAdjustment()
    clf.fit(X, y, z, c_ft_value=c)
    return clf

def show_most_correlated_features(train_x, train_corr, correlate, feature_names, perc=0.2):
    """This class is adapted from github.com/tapilab/aaai-2016-robust"""

    # add a star when printing one of the top correlated feature in the feature with the highest coeffs
    x2, pval = chi2(train_x, train_corr)
    n_top_corr_fts = round(train_x.shape[1]*perc/100)
    top_ft_idx = numpy.argsort(x2)[::-1][:n_top_corr_fts]
    top_ft_names = [feature_names[i] for i in top_ft_idx]
    print("Top " + str(perc*100) + "% of features that are the most correlated with " + correlate + ":\n" + ', '.join(top_ft_names))
    return top_ft_idx, top_ft_names

def get_features_with_most_coef_change(model, vectorizer, baseline_model, confound):
    print('Features with greatest Coeficient Change')
    features = dict(zip(vectorizer.feature_names_, model.clf.coef_[0]))
    baseline_features = dict(zip(vectorizer.feature_names_, baseline_model.coef_[0]))
    diffs = []
    for feat in baseline_features:
        diffs.append((feat, abs(features[feat]-baseline_features[feat])))
    diffs.sort(key=lambda x: x[1],reverse=True)
    stronger_coefs = []
    weaker_coefs = []
    i = 0
    while (len(stronger_coefs) < 10 or len(weaker_coefs) < 10) and i < len(baseline_features):
        if abs(features[diffs[i][0]]) < abs(baseline_features[diffs[i][0]]):
            if len(weaker_coefs) < 10:
                weaker_coefs.append(diffs[i])
        elif abs(features[diffs[i][0]]) > abs(baseline_features[diffs[i][0]]):
            if len(stronger_coefs) < 10:
                stronger_coefs.append(diffs[i])
        i += 1
    print('features that are getting WEAKER with confounding adjustment')

    for feat in weaker_coefs:
        print(feat[0] + '\t diff: ' + str(feat[1]) + '\t baseline: ' + str(
            baseline_features[feat[0]]) + '\t backdoor: ' + str(features[feat[0]]))
    print('features that are getting STRONGER with confounding adjustment')
    for feat in stronger_coefs:
        print(feat[0] + '\t diff: ' + str(feat[1]) + '\t baseline: ' + str(
            baseline_features[feat[0]]) + '\t backdoor: ' + str(features[feat[0]]))

    plot_coefs(baseline_features,features,weaker_coefs,'weakened',confound)
    plot_coefs(baseline_features,features,stronger_coefs,'strengthened',confound)


def plot_coefs(baseline_features, features, coefs, change, confound):
    value1 = [baseline_features[x[0]] for x in coefs]
    value2 = [features[x[0]] for x in coefs]
    myrange = range(0,len(value1))
    pyplot.figure(figsize=(10,8))
    pyplot.hlines(y=myrange, xmin=value1, xmax=value2, color='black', alpha=1)
    pyplot.scatter(value1, myrange, color='blue', alpha=1, label='baseline')
    pyplot.scatter(value2, myrange, color='green', alpha=1, label='backdoor')

    feat_names = [x[0] for x in coefs]
    pyplot.yticks(myrange, feat_names,fontweight='demibold')
    pyplot.xticks(numpy.arange(-.8,.9,.1),fontweight='demibold')
    if confound == '1':
        conf_name = 'Inpatient Notes'
    elif confound == '2':
        conf_name = 'Outpatient Notes'
    elif confound == 'PCC' or confound == 'PICSI' or confound == 'FCS':
        conf_name = confound + ' Dataset'
    elif confound == 'EXCLUDE':
        conf_name = 'Other'
    elif confound == 'DCSUMMARY':
        conf_name = 'Discharge Summary'
    elif confound == 'ADMITNOTE':
        conf_name = 'Admit'
    elif confound == 'PN':
        conf_name = 'Progress'
    elif confound == 'ED':
        conf_name = 'Emergency Department'
    elif confound == 'CODESTATUS':
        conf_name = 'Code Status'
    elif confound == 'SW':
        conf_name = 'Social Work'
    elif confound == 'NURSING':
        conf_name = 'Nursing'
    elif confound == 'OTHERSUMMERY':
        conf_name = 'Other Summary'
    else:
        conf_name = confound.lower().capitalize() + ' Notes'

    pyplot.title('Most ' + change + ' features when adjusting for ' + conf_name,fontweight='demibold')
    pyplot.xlabel('Feature Coeficient',fontweight='demibold')
    pyplot.ylabel('Feature Name',fontweight='demibold')
    pyplot.legend()
    pyplot.show()

def create_false_positive_graph(false_positives):
    fig = pyplot.figure(figsize=(10,10))
    pyplot.yticks(fontweight='demibold')
    pyplot.xticks(numpy.arange(0,12,1), fontweight='demibold')
    ax = fig.add_subplot(1,1,1)
    note_id = 1
    for fp in false_positives:
        if 'baseline' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['baseline'],color='black',label='Baseline',alpha=1,marker=',',s=80)
        if '1' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['1'],color='blue',label='In/Outpatient',alpha=1,marker='>',s=60)
        if 'fcs' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['fcs'],color='yellow',label='FCS',alpha=1,marker='^',s=60)
        if 'pcc' in false_positives[fp]:
            print(fp+'\tpcc')
            ax.scatter(note_id,false_positives[fp]['pcc'],color='green',label='PCC',alpha=1,marker='^',s=60)
        if 'picsi' in false_positives[fp]:
            print(fp + '\tpicsi')
            ax.scatter(note_id,false_positives[fp]['picsi'],color='red',label='PICSI',alpha=1,marker='^')
        if 'admitnote' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['admitnote'],color='#008000',label='Admit Note',alpha=1,marker='.')
        if 'pn' in false_positives[fp]:
            ax.scatter(note_id, false_positives[fp]['pn'], color='#800080', label='Progress Note', alpha=1,marker='.')
        if 'dcsummary' in false_positives[fp]:
            ax.scatter(note_id, false_positives[fp]['dcsummary'], color='#00FFFF', label='Discharge Summary', alpha=1,marker='.')
        if 'othersummary' in false_positives[fp]:
            ax.scatter(note_id, false_positives[fp]['othersummary'], color='#008080', label='Other Summary',alpha=1,marker='.')
        if 'ed' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['ed'],color='#000080',label='Emergency Department',alpha=1,marker='.')
        if 'codestatus' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['ed'],color='#000080',label='Code Status',alpha=1,marker='.')
        if 'nursing' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['nursing'],color='#800080',label='Nursing',alpha=1,marker='.')

        if 'sw' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['sw'],color='#00FF00',label='Social Work',alpha=1,marker='.')
        if 'exclude' in false_positives[fp]:
            ax.scatter(note_id,false_positives[fp]['exclude'],color='#FF00FF',label='Other',alpha=1,marker='.')
        note_id += 1
    handles, labels = pyplot.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))
    pyplot.legend(by_label.values(), by_label.keys())
    pyplot.title('Prediction Strength for False Positives', fontweight='demibold')
    pyplot.xlabel('Note ID', fontweight='demibold')
    pyplot.ylabel('Prediction', fontweight='demibold')
    pyplot.show()

def get_top_features(vectorizer, model):
    feature_names = vectorizer.feature_names_
    top10 = numpy.argsort(model.coef_[0])[-20:]
    bottom10 = numpy.argsort(model.coef_[0])[:20]
    print("%s: %s" % ('pos', ", ".join(feature_names[j] for j in top10)))
    print('\n')
    # print("%s: %s" % ('neg', "\n".join(feature_names[j] for j in bottom10)))
    # print('\n')
    #return [feature_names[j] for j in top10], [feature_names[j] for j in bottom10]

def print_results(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, output_dict=True,digits=4)
    print(report)
    return report['1']

def plot_results(results):
    for model in results:
        results[model]['f0.5-score'] = (1.25*results[model]['precision']*results[model]['recall'])/((0.25*results[model]['precision'])+results[model]['recall'])
    for metric in ['precision', 'recall', 'f1-score', 'f0.5-score']:
        fig = pyplot.figure(figsize=(10,5))
        pyplot.yticks(fontweight='demibold')
        pyplot.xticks(fontweight='demibold')
        pyplot.axhline(y=results['baseline'][metric],color='black',linestyle='-')
        ax = fig.add_subplot(1,1,1)
        ax.scatter('Inpatient',results['1'][metric],color='blue',alpha=1,marker='*')
        ax.scatter('Outpatient', results['2'][metric], color='blue', alpha=1, marker='*')
        ax.scatter('PCC', results['PCC'][metric], color='green', alpha=1, marker='*')
        ax.scatter('PICSI', results['PICSI'][metric], color='green', alpha=1, marker='*')
        ax.scatter('FCS', results['FCS'][metric], color='green', alpha=1, marker='*')
        ax.scatter('Admit', results['ADMITNOTE'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Progress', results['PN'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Discharge Summary', results['DCSUMMARY'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Other Summary', results['OTHERSUMMARY'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Emergency Department', results['ED'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Code Status', results['CODESTATUS'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Nursing', results['NURSING'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Social Work', results['SW'][metric], color='purple', alpha=1, marker='*')
        ax.scatter('Other Note', results['EXCLUDE'][metric], color='purple', alpha=1, marker='*')

        pyplot.title('Results on Test Data:' + metric.capitalize()+' for each model, compared to the baseline', fontweight='demibold')
        pyplot.xlabel('Model', fontweight='demibold')
        pyplot.ylabel(metric.capitalize(), fontweight='demibold')
        pyplot.xticks(rotation=60)
        pyplot.show()

def get_average(scores):
    average = numpy.mean(scores)
    stderr = stats.sem(scores)
    return average, stderr

def plot_cross_val_results(results):
    for model in results:
        for split in results[model]:
            results[model][split]['f0.5-score'] = (1.25*results[model][split]['precision']*results[model][split]['recall'])/((0.25*results[model][split]['precision'])+results[model][split]['recall'])
    for metric in ['precision', 'recall', 'f1-score', 'f0.5-score']:
        fig = pyplot.figure(figsize=(10,5))
        pyplot.yticks(fontweight='demibold')
        pyplot.xticks(fontweight='demibold')
        avg, err = get_average([results['baseline'][x][metric] for x in results['baseline']])
        pyplot.axhline(y=avg,color='black',linestyle='solid')
        pyplot.axhline(y=avg+err, color='black', linestyle='dashed')
        pyplot.axhline(y=avg-err, color='black', linestyle='dashed')
        ax = fig.add_subplot(1,1,1)
        avg, err = get_average([results['1'][x][metric] for x in results['1']])
        ax.scatter('Inpatient',avg,color='blue',alpha=1,marker='*')
        ax.scatter('Inpatient',avg+err, color='red', alpha=1, marker='*')
        ax.scatter('Inpatient',avg-err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['2'][x][metric] for x in results['2']])
        ax.scatter('Outpatient',avg,color='blue',alpha=1,marker='*')
        ax.scatter('Outpatient',avg+err, color='red', alpha=1, marker='*')
        ax.scatter('Outpatient',avg-err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['PCC'][x][metric] for x in results['PCC']])
        ax.scatter('PCC',avg,color='green',alpha=1,marker='*')
        ax.scatter('PCC',avg+err, color='red', alpha=1, marker='*')
        ax.scatter('PCC',avg-err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['PICSI'][x][metric] for x in results['PICSI']])
        ax.scatter('PICSI', avg, color='green', alpha=1, marker='*')
        ax.scatter('PICSI', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('PICSI', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['FCS'][x][metric] for x in results['FCS']])
        ax.scatter('FCS', avg, color='green', alpha=1, marker='*')
        ax.scatter('FCS', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('FCS', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['ADMITNOTE'][x][metric] for x in results['ADMITNOTE']])
        ax.scatter('Admit', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Admit', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Admit', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['PN'][x][metric] for x in results['PN']])
        ax.scatter('Progress', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Progress', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Progress', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['DCSUMMARY'][x][metric] for x in results['DCSUMMARY']])
        ax.scatter('Discharge Summary', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Discharge Summary', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Discharge Summary', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['OTHERSUMMARY'][x][metric] for x in results['OTHERSUMMARY']])
        ax.scatter('Other Summary', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Other Summary', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Other Summary', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['ED'][x][metric] for x in results['ED']])
        ax.scatter('Emergency Department', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Emergency Department', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Emergency Department', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['CODESTATUS'][x][metric] for x in results['CODESTATUS']])
        ax.scatter('Code Status', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Code Status', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Code Status', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['NURSING'][x][metric] for x in results['NURSING']])
        ax.scatter('Nursing', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Nursing', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Nursing', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['SW'][x][metric] for x in results['SW']])
        ax.scatter('Social Work', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Social Work', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Social Work', avg - err, color='red', alpha=1, marker='*')
        avg, err = get_average([results['EXCLUDE'][x][metric] for x in results['EXCLUDE']])
        ax.scatter('Other Note', avg, color='purple', alpha=1, marker='*')
        ax.scatter('Other Note', avg + err, color='red', alpha=1, marker='*')
        ax.scatter('Other Note', avg - err, color='red', alpha=1, marker='*')

        pyplot.title('Cross Validation Results over full dataset: ' + metric.capitalize()+' for each model, compared to the baseline', fontweight='demibold')
        pyplot.xlabel('Model', fontweight='demibold')
        pyplot.ylabel(metric.capitalize(), fontweight='demibold')
        pyplot.xticks(rotation=60)
        pyplot.show()

def build_featuresets(notes):
    featuresets = []
    for i in range(0, len(notes)):
        doc_features = notes[i][FEATURE_INDEX].split(', ')
        doc_featureset = {}
        # Create a featureset (a dictionary with each features and the number of times it occurs in the document),
        # excluding any numbers or empty strings
        deduped_features = {}
        for feat in doc_features:
            if any(char.isdigit() for char in feat):
                continue
            if feat == '':
                continue
            if feat not in doc_featureset:
                doc_featureset[feat] = 1
            else:
                doc_featureset[feat] += 1
            if feat not in deduped_features:
                deduped_features[feat] = 1
        note_metadata = {str(INPATIENT_INDEX): notes[i][INPATIENT_INDEX],
                         str(DATASET_INDEX): notes[i][DATASET_INDEX],
                         str(NOTE_TYPE_INDEX): notes[i][NOTE_TYPE_INDEX]}

        featuresets.append([deduped_features, notes[i][2], note_metadata])
    return featuresets

def print_confound_goc_distribution(dataset, confounders):
    print(str(len(dataset)))
    for index, confounder in confounders:
        goc_pos_count = 0
        goc_neg_count = 0
        confound_count = 0
        for note in dataset:
            if confounder[0] in str(note[index]):
                confound_count += 1
                if note[GOC_INDEX] == 'pos':
                    goc_pos_count += 1
                else:
                    goc_neg_count += 1
        print(confounder[0] + ' notes:\nTotal: ' + str(confound_count) + '\nGOC\+: ' + str(goc_pos_count) + '\nGOC-: ' + str(goc_neg_count))
    print('\n')

def run_baseline(train_featuresets, test_featuresets, test_notes):
    print('Baseline ')
    train_vect = DictVectorizer().fit([i[0] for i in train_featuresets])
    train_x = train_vect.transform([i[0] for i in train_featuresets])
    train_y = [1 if i[1] == 'pos' else 0 for i in train_featuresets]
    baseline_model = LogisticRegression(class_weight='balanced')
    baseline_model.fit(train_x, train_y)
    predictions = []
    decisions = []
    gold = []
    false_positives = {}
    for i in range(0, len(test_featuresets)):
        prediction = baseline_model.predict(train_vect.transform(test_featuresets[i][0]))
        decision = baseline_model.predict_proba(train_vect.transform(test_featuresets[i][0]))[0][1]
        predictions.append(prediction[0])
        decisions.append(decision)
        true_val = 1 if test_featuresets[i][1] == 'pos' else 0
        gold.append(true_val)
        if test_notes != []:
            if prediction[0] == 1 and true_val == 0:
                false_positives[test_notes[i][0]] = {'baseline': decision}
                print(test_notes[i][0] + ':\t' + str(decision))

    return gold, predictions, false_positives

def do_backdoor_adjustment(train_featuresets, train_x, train_y, train_vect, confound_index, confound_values, baseline_model, test_featuresets, test_notes, false_positives):
    predictions = []
    gold = []
    decisions = []

    train_c = [1 if confound_values[0] in str(i[2][str(confound_index)]) else 0 for i in train_featuresets]
    print(str(len([x for x in train_c if x == 1])) + '/' + str(len(train_c)))
    backdoor_model = backdoor_adjustment_var_C(train_x, train_y, train_c, 1.)
    train_vect.feature_names_.append(confound_values[0])#I'm not sure if this is okay. As long as we only use this to look map features back to their names, it should be okay
    train_vect.feature_names_.append('not_'+confound_values[0])
    for i in range(0, len(test_featuresets)):
        prediction, decision = backdoor_model.predict(train_vect.transform(test_featuresets[i][0]))
        predictions.append(prediction)
        decisions.append(decision)
        true_val = 1 if test_featuresets[i][1] == 'pos' else 0
        gold.append(true_val)
        if test_notes != []:
            if prediction == 1 and true_val == 0:
                if test_notes[i][0] in false_positives:
                    false_positives[test_notes[i][0]][confound_values[0].lower()] = decision
                else:
                    false_positives[test_notes[i][0]] = {confound_values[0].lower(): decision}
                print(test_notes[i][0] + ':\t' + str(decision))
    get_features_with_most_coef_change(backdoor_model, train_vect, baseline_model, confound_values[0])

    pyplot.show()
    results = print_results(gold, predictions, classes=[0,1])

    return false_positives, gold, predictions, results


database = '../merged_splits.db'
conn = sqlite3.connect(database)
curs = conn.cursor()
train_notes = curs.execute("SELECT * FROM TRAINING_SET").fetchall()
dev_notes = curs.execute("SELECT * FROM DEV_SET").fetchall()
test_notes = curs.execute("SELECT * FROM TEST_SET").fetchall()
conn.close


train_featuresets = build_featuresets(train_notes)
dev_featuresets = build_featuresets(dev_notes)
test_featuresets = build_featuresets(test_notes)

confounders = [[INPATIENT_INDEX, ['1','2']],[INPATIENT_INDEX, ['2','1']],[DATASET_INDEX, ['PCC','not_PCC']],
               [DATASET_INDEX, ['PICSI','not_PICSI']],[DATASET_INDEX, ['FCS','not_FCS']],
               [NOTE_TYPE_INDEX, ['NURSING','not_NURSING']],
               [NOTE_TYPE_INDEX, ['ADMITNOTE','not_ADMITNOTE']], [NOTE_TYPE_INDEX, ['EXCLUDE','not_EXCLUDE']],
               [NOTE_TYPE_INDEX, ['CODESTATUS','not_CODESTATUS']], [NOTE_TYPE_INDEX, ['DCSUMMARY','not_DCSUMMARY']],
               [NOTE_TYPE_INDEX, ['ED','not_ED']], [NOTE_TYPE_INDEX, ['PN','not_PN']], [NOTE_TYPE_INDEX, ['SW','not_SW']],
               [NOTE_TYPE_INDEX, ['OTHERSUMMARY','not_OTHERSUMMARY']]]

gold, predictions, false_positives = run_baseline(train_featuresets, test_featuresets, test_notes)
baseline_result = print_results(gold, predictions, classes=[0, 1])
results = {'baseline': baseline_result}

# Do cross validation:
full_featuresets = train_featuresets + dev_featuresets + test_featuresets
random.shuffle(full_featuresets)
size_fold = round(len(full_featuresets) / 10)
g = []
p = []
cross_val_results = {}
for i in range(1, 11):
    k_test_featuresets = full_featuresets[(i - 1) * size_fold:i * size_fold]
    k_train_featuresets = full_featuresets[:(i - 1) * size_fold] + full_featuresets[i * size_fold:]
    gold, predictions, k_false_positives = run_baseline(k_train_featuresets, k_test_featuresets, [])
    g.extend(gold)
    p.extend(predictions)
    cross_val_results[i] = print_results(gold, predictions, classes=[0, 1])

print("K-FOLD RESULTS")  # This will be result across full cv
kfold_results = {'baseline': cross_val_results}

train_vect = DictVectorizer().fit([i[0] for i in train_featuresets])
train_x = train_vect.transform([i[0] for i in train_featuresets])
train_y = [1 if i[1] == 'pos' else 0 for i in train_featuresets]
baseline_model = LogisticRegression(class_weight='balanced')
baseline_model.fit(train_x, train_y)
for index, values in confounders:
    print(values[0])
    false_positives, gold, prediction, model_results = do_backdoor_adjustment(train_featuresets, train_x, train_y, train_vect, index, values, baseline_model, test_featuresets, test_notes, false_positives)
    results[str(values[0])] = model_results
    cross_val_results = {}

    for i in range(1, 11):
        k_test_featuresets = full_featuresets[(i - 1) * size_fold:i * size_fold]
        k_train_featuresets = full_featuresets[:(i - 1) * size_fold] + full_featuresets[i * size_fold:]
        k_train_vect = DictVectorizer().fit([i[0] for i in k_train_featuresets])
        k_train_x = k_train_vect.transform([i[0] for i in k_train_featuresets])
        k_train_y = [1 if i[1] == 'pos' else 0 for i in k_train_featuresets]
        k_baseline_model = LogisticRegression(class_weight='balanced')
        k_baseline_model.fit(k_train_x, k_train_y)
        k_false_positives, gold, predictions, model_results = do_backdoor_adjustment(k_train_featuresets, k_train_x, k_train_y, k_train_vect, index, values, k_baseline_model, k_test_featuresets, [], {})
        cross_val_results[i] = print_results(gold,predictions, classes=[0,1])

    print("K-FOLD RESULTS", values[0])  # Where full cv results will be
    kfold_results[str(values[0])] = cross_val_results
    y_corr_idx, y_corr_names = show_most_correlated_features(train_x, train_y, 'goals of care', train_vect.feature_names_, perc=0.1)
    get_top_features(train_vect, baseline_model)

    print_results(gold, predictions, classes=[0,1])

plot_results(results)
plot_cross_val_results(kfold_results)
create_false_positive_graph(false_positives)
print('Train Notes:')
print_confound_goc_distribution(train_notes, confounders)
print('Dev Notes:')
print_confound_goc_distribution(dev_notes, confounders)
print('Test Notes:')
print_confound_goc_distribution(test_notes, confounders)
