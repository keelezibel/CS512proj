import csv
import numpy
import pandas
import time
import argparse
import constants

def read_data(datafile):
    return pandas.read_csv(
        datafile, sep='\t', index_col=False)


def read_x(filename, savedir):
    return pandas.read_csv(
        savedir + '/' + filename + '.csv',
        sep='\t', index_col=0, header=None)


def create_matrixRX(repo, x):
    matrixRX = pandas.DataFrame(numpy.zeros(shape=(len(repo), len(x))))
    matrixRX = matrixRX.set_index(repo.index.values)
    matrixRX.columns = x.index.values
    return matrixRX


def set_relation(matrixRX, type_x, datafile):
    data = read_data(datafile)

    for index, row in data.iterrows():
        cur_repo = row[constants.REPO]
        cur_x = row[type_x]
        if type == constants.LANG:
            # make sure a repository gets one language only once
            matrixRX.loc[cur_repo][cur_x] = 1
        else:
            matrixRX.loc[cur_repo][cur_x] += 1


def create_matrices(actor, lang, repo, datafile, savedir):
    try:
        # load matrix from csv
        matrixRA = pandas.DataFrame.from_csv(
            savedir + '/' + 'matrixRA.csv')
        matrixRL = pandas.DataFrame.from_csv(
            savedir + '/' + 'matrixRL.csv')

    except IOError:
        matrixRA = create_matrixRX(repo, actor)
        matrixRL = create_matrixRX(repo, lang)

        set_relation(matrixRA, constants.ACTOR, datafile)
        set_relation(matrixRL, constants.LANG, datafile)

        # save matrix to csv
        matrixRA.to_csv(savedir + '/' + 'matrixRA.csv')
        matrixRL.to_csv(savedir + '/' + 'matrixRL.csv')

    return matrixRA, matrixRL


def create_matrixARA(matrixRA, savedir):
    try:
        # load matrix from csv
        matrixARA = pandas.DataFrame.from_csv(
            savedir + '/' + 'matrixARA.csv')
    except IOError:
        matrixAR = matrixRA.transpose()
        matrixARA = matrixAR.dot(matrixRA)

        # save matrix to csv
        matrixARA.to_csv(savedir + '/' + 'matrixARA.csv')

    return matrixARA


def create_matrixARLRA(matrixRL, matrixRA, savedir):
    try:
        # load matrix from csv
        matrixARLRA = pandas.DataFrame.from_csv(
            savedir + '/' + 'matrixARLRA.csv')
    except IOError:
        matrixAR = matrixRA.transpose()
        matrixARL = matrixAR.dot(matrixRL)
        matrixLRA = matrixARL.transpose()
        matrixARLRA = matrixARL.dot(matrixLRA)

        # save matrix to csv
        matrixARLRA.to_csv(savedir + '/' + 'matrixARLRA.csv')
    return matrixARLRA


def top_k_similar(actor_name, k, matrix):
    similar = matrix.loc[actor_name].copy()
    for other_actor in similar.index:
        similar[other_actor] = (
            2 * matrix.loc[actor_name][other_actor] / (
                matrix.loc[actor_name][actor_name] +
                matrix.loc[other_actor][other_actor]))

    similar.sort_values(inplace=True, ascending=False)

    return similar[0:k]


def get_test_result(k, matrixAXA, test_actors):
    temp_results = []

    for test_actor, _ in test_actors.iterrows():
        top_k = top_k_similar(test_actor, k, matrixAXA)
        temp_results.append(top_k.index.tolist())

    resultAXA = pandas.DataFrame(temp_results)
    resultAXA = resultAXA.set_index(test_actors.index.values)
    return resultAXA

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("datafile")
parser.add_argument("traindir")
parser.add_argument("testdir")
args = parser.parse_args()

# read in output files of list_all_x.py
langs = read_x(constants.LANG, args.traindir)
actors = read_x(constants.ACTOR, args.traindir)
repos = read_x(constants.REPO, args.traindir)

test_actors = read_x(constants.ACTOR, args.testdir)

matrixRA, matrixRL = create_matrices(
    actors, langs, repos, args.datafile, args.traindir)

# compute result using meta-path ARA
matrixARA = create_matrixARA(matrixRA, args.traindir)
resultARA = get_test_result(5, matrixARA, test_actors)
resultARA.to_csv('pathsimARA_5.csv', header=False)
resultARA = get_test_result(10, matrixARA, test_actors)
resultARA.to_csv('pathsimARA_10.csv', header=False)
resultARA = get_test_result(15, matrixARA, test_actors)
resultARA.to_csv('pathsimARA_15.csv', header=False)


# compute result using meta-path ARLRA
matrixARLRA = create_matrixARLRA(matrixRL, matrixRA, args.traindir)
resultARLRA = get_test_result(5, matrixARLRA, test_actors)
resultARLRA.to_csv('pathsimARLRA_5.csv', header=False)
resultARLRA = get_test_result(10, matrixARLRA, test_actors)
resultARLRA.to_csv('pathsimARLRA_10.csv', header=False)
resultARLRA = get_test_result(15, matrixARLRA, test_actors)
resultARLRA.to_csv('pathsimARLRA_15.csv', header=False)
