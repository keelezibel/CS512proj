import pandas
import numpy
import argparse

import constants


def read_x(filename, savedir):
    return pandas.read_csv(
        savedir + '/' + filename + '.csv',
        sep='\t', index_col=0, header=None)


def get_adjacency(metapath, traindir):
    matrixAXA = pandas.DataFrame.from_csv(
        traindir + '/' + 'matrix' + metapath + '.csv')
    matrixAXA[matrixAXA != 0] = 1
    return matrixAXA


def create_adjacency_matrix(metapath, actors, traindir):
    matrix_csv = 'adjacency_matrix_for_' + metapath + '.csv'

    try:
        adjacency = pandas.DataFrame.from_csv(matrix_csv)
        adjacency.columns = adjacency.index.values
        # print('matrix is loaded')
    except IOError:
        # initialize matrix AXA
        # adjacency = pandas.DataFrame(
        #     numpy.identity(len(actors)))
        # adjacency = adjacency.set_index(actors.index.values)
        # adjacency.columns = actors.index.values

        adjacency = get_adjacency(metapath, traindir)

        # normalize the matrix
        row_sum = adjacency.sum(axis=1)
        adjacency = adjacency.div(row_sum, axis='rows')

        adjacency.to_csv(matrix_csv)
        # print('matrix is saved')

    print('\nadjacency:\n', adjacency)
    return adjacency


def p_pagerank(test_actor, adjacency, actors):
    # create preference vector u
    preference = pandas.DataFrame(
        numpy.zeros(shape=(len(actors), 1)))
    preference = preference.set_index(actors.index.values)
    preference.loc[test_actor] = 1

    # initialize scores vector v
    score = pandas.DataFrame(
        numpy.ones(shape=(len(actors), 1)))
    score = score / len(actors)
    score = score.set_index(actors.index.values)

    # iterate
    t = constants.ITERATION_TIME
    while(t > 0):
        t -= 1
        score = adjacency.dot(score) * (1 - constants.TELEPORT) + (
            constants.TELEPORT * preference)

    return score


def top_k_similar(test_actor, k, adjacency, actors):
    similar = p_pagerank(test_actor, adjacency, actors)
    similar = similar.squeeze()
    similar = similar.copy()

    similar.sort_values(inplace=True, ascending=False)
    return similar[0:k]


def get_test_result(k, adjacencyAXA, test_actors, actors):
    temp_results = []

    for test_actor, _ in test_actors.iterrows():
        top_k = top_k_similar(test_actor, k, adjacencyAXA, actors)
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

actors = read_x(constants.ACTOR, args.traindir)
test_actors = read_x(constants.ACTOR, args.testdir)

adjacencyARA = create_adjacency_matrix('ARA', actors, args.traindir)
adjacencyARLRA = create_adjacency_matrix('ARA', actors, args.traindir)

# compute result using meta-path ARA
resultARA = get_test_result(5, adjacencyARA, test_actors, actors)
resultARA.to_csv('ppagerankARA_5.csv', header=False)
