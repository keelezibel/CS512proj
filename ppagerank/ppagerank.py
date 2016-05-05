import pandas
import numpy
import csv

import constants


def read_author():
    return pandas.read_csv(
        constants.datapath + constants.AUTHOR + '.txt',
        sep='\t', index_col=0, header=None)


def parse_link(matrix, metapath):
    with open(constants.datapath + metapath + '.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        for row in reader:
            aid1, aid2 = row
            aid1 = int(aid1)
            aid2 = int(aid2)

            matrix.loc[aid1, aid2] += 1
            matrix.loc[aid2, aid1] += 1

            if matrix.loc[aid1, aid2] > 1:
                print('m[', aid1, ', ', aid2, '] = ', matrix.loc[aid1, aid2])


def create_adjacency_matrix(metapath, author):
    filepath = constants.datapath + 'adjacency_matrix_for_' + metapath + '.csv'

    try:
        adjacency = pandas.DataFrame.from_csv(filepath)
        adjacency.columns = adjacency.index.values
        # print('matrix is loaded')
    except IOError:
        # initialize
        adjacency = pandas.DataFrame(
            numpy.identity(len(author)))
        adjacency = adjacency.set_index(author.index.values)
        adjacency.columns = author.index.values

        parse_link(adjacency, metapath)

        # normalize the matrix
        row_sum = adjacency.sum(axis=1)
        adjacency = adjacency.div(row_sum, axis='rows')

        adjacency.to_csv(filepath)
        # print('matrix is saved')

    print('\nadjacency:\n', adjacency)
    return adjacency


def p_pagerank(qid, adjacency, author):
    # create preference vector u
    preference = pandas.DataFrame(
        numpy.zeros(shape=(len(author), 1)))
    preference = preference.set_index(author.index.values)
    preference.loc[qid] = 1

    # initialize scores vector v
    score = pandas.DataFrame(
        numpy.ones(shape=(len(author), 1)))
    score = score / len(author)
    score = score.set_index(author.index.values)

    # iterate
    t = constants.ITERATION_TIME
    while(t > 0):
        t -= 1
        score = adjacency.dot(score) * (1 - constants.TELEPORT) + (
            constants.TELEPORT * preference)

    return score


def top_k_similar(qid, k, adjacency, author):
    similar = p_pagerank(qid, adjacency, author)
    similar = similar.squeeze()
    similar = similar.copy()

    similar.sort_values(inplace=True, ascending=False)
    return similar[0:k]


def print_result(result, author):
    for similar_aid, score in result.iteritems():
        print(similar_aid, '\t', author.loc[similar_aid][1], '\t', score)


author = read_author()

print('\nadjacency matrix for APVPA is created:')
adjacencyAPVPA = create_adjacency_matrix('APVPA', author)

print('\nThe top similar authors to A. Apple using APVPA are:\n')
result = top_k_similar(42166, 5, adjacencyAPVPA, author)
print_result(result, author)
