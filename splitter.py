import numpy as np
import pandas as pd


def sortDataset():
    pd.read_csv("train.csv").sort_values(by=['timestamp']).to_csv("sortedTrain.csv", index=False)


def splitSet(threshold):
    df = pd.read_csv("sortedTrain.csv", nrows=100000)
    minTimestamp = df['timestamp'].min()
    maxTimestamp = df['timestamp'].max()
    thresholdTimestamp = minTimestamp + (maxTimestamp - minTimestamp)*threshold

    print("Threshold timestamp: " + str(thresholdTimestamp))

    print("Scan set")
    trainSetSessions = df.copy().loc[df['timestamp'] <  thresholdTimestamp]['session_id'].values.tolist()
    testSetSessions  = df.copy().loc[df['timestamp'] >= thresholdTimestamp]['session_id'].values.tolist()

    print("Create set")
    trainDf = df.copy().loc[df['session_id'].isin(trainSetSessions)]
    testDf  = df.copy().loc[df['session_id'].isin(set(testSetSessions) - set(trainSetSessions))]

    print("Used rows: " + str((trainDf.shape[0] + testDf.shape[0])/df.shape[0]*100) + "%")
    print("Trainset shape: " + str(trainDf.shape))
    print("Testset shape: " + str(testDf.shape))

    print("Save ground truth")
    testDf.to_csv("splitterOut/gt.csv", index=False)

    print('Update reference values to nan')
    testDf['index'] = range(0, testDf.shape[0])
    indices = testDf \
        .copy() \
        .loc[lambda df: df['action_type'] == 'clickout item'] \
        .groupby(['user_id'], as_index=False) \
        .max(level='timestamp')['index'].values.tolist()
    testDf.loc[lambda df: df['index'].isin(indices), 'reference'] = np.nan
    del testDf['index']

    print("Updated " + str(len(indices)) + " reference values to np.nan")

    print("Save train set")
    trainDf.to_csv("splitterOut/train.csv", index=False)
    print("Save test set")
    testDf.to_csv("splitterOut/test.csv", index=False)


if __name__ == "__main__":
    splitSet(0.92)