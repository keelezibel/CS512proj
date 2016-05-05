import argparse
import pandas
import numpy


def create_matrixTA(test_actors, matrixARA):
    matrixTA = pandas.DataFrame(
        numpy.zeros(shape=(len(test_actors), len(matrixARA))))
    matrixTA = matrixTA.set_index(test_actors.index.values)
    matrixTA.columns = matrixARA.index.values
    return matrixTA


def get_matrixTA(filepath):
    matrixTA = pandas.DataFrame.from_csv(filepath, header=None)
    return matrixTA


def get_matrixARA(filepath):
    matrixARA = pandas.DataFrame.from_csv(filepath)
    return matrixARA

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("test_result_csv")
parser.add_argument("matrixARA_csv")
args = parser.parse_args()


matrixTA = get_matrixTA(args.test_result_csv)
matrixARA = get_matrixARA(args.matrixARA_csv)
# print(matrixTA)

eval_result = pandas.DataFrame(
    numpy.zeros(shape=matrixTA.shape, dtype=bool),
    columns=numpy.arange(1, matrixTA.shape[1]+1))
eval_result = eval_result.set_index(matrixTA.index.values)
# print(eval_result)

total_simialrs = matrixTA.shape[0] * matrixTA.shape[1]
correct_similars = 0
for test_actor, similars in matrixTA.iterrows():
    # print('row:', similars)
    for k, similar in similars.iteritems():
        # print('similar:', similar)
        try:
            if matrixARA.loc[test_actor][similar] > 0:
                eval_result.loc[test_actor][k] = True
                correct_similars += 1
        except KeyError:
            pass  # print(similar, "doesn't exist in matrixARA")

print(
    "\nFor ", args.test_result_csv, ":\n",
    "Out of your total ", total_simialrs, "predictions, ",
    correct_similars, "of them are indeed co-commiters. ")
print("That's ", correct_similars / total_simialrs * 100, "percent corrent. ")
print("(See eval_result.csv to know which of them you got wrong. )")

# print(eval_result)
eval_result.to_csv('eval_result.csv', header=False)
