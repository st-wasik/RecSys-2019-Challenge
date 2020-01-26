import string

import pandas as pd
import math
import pandas as pd
import numpy as np


GR_COLS = ["user_id", "session_id", "timestamp", "step"]

def fastSplitting(df): #stack overflow ❤
    df2 = df.loc[df["properties"]]
    s = df2["properties"].str.split('|', expand=True).stack()
    i = s.index.get_level_values(0)
    df3 = df.loc[i].copy()
    df3["properties"] = s.values
    return df3

def getPopularImpressions():
    train = pd.read_csv("sortedTrain.csv")

    references = train.loc[train['action_type']=='clickout item'].groupby("reference")\
        .size().to_frame().reset_index().transform(lambda x: x.astype(int))
    references.columns = ['reference', 'clicks']
    print(references)

    occurences = explode(train, "impressions").groupby("impressions")\
        .size().to_frame().reset_index().transform(lambda x: x.astype(int))
    occurences.columns = ['reference', 'occurences']
    print(occurences)

    popularity = pd.merge(references, occurences, on='reference')
    popularity['popularity'] = popularity['clicks'] / popularity['occurences']
    popularity = popularity[['reference','popularity']]
    popularity['popularity'] = np.where(popularity['popularity'] >= 1.0, 0, popularity['popularity'])
    popularity = popularity.sort_values('popularity', ascending=False)
    popularity = popularity[:20]
    print(popularity)

    metadata = pd.read_csv("splitterOut/item_metadata.csv")
    metadata['properties'] = metadata['properties'].str.replace(' ', '_')

    traits = popularity.merge(metadata, left_on='reference', right_on='item_id')
    print(traits)
    traits = explode(traits, "properties", False)\
        .groupby("properties", as_index=False)\
        .size().to_frame().reset_index()
    traits.columns = ['properties', 'count']
    traits['count'] = np.where(traits['count'] >= 7, traits['count'], 0)
    traits = traits.loc[traits['count'] != 0]
    traits = traits.sort_values(by='count', ascending=False)
    print(traits)
    traits_list = traits['properties'].values.tolist()
    print(traits_list)

    print('Convert properties to traits')
    for i in traits_list:
        metadata[i] = np.where(
            metadata.properties.str.contains('|' + i + '|', regex=False), 1, 0)
    print('Drop properties')
    del metadata["properties"]
    print('Save metadata to file')
    metadata.to_csv("splitterOut/hotelTraits.csv", index=False)

def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )
    return df_item_clicks


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expl, applyInt=True):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    if applyInt:
        df_out.loc[:, col_expl] = df_out[col_expl].apply(int)
    else:
        df_out.loc[:, col_expl] = df_out[col_expl]

    return df_out


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out

def calc_recommendation(df_target, meta_data):

    df_target["impressions"] = df_target["impressions"].astype(int)
    meta_data["item_id"] = meta_data["item_id"].astype(int)
    df_expl_clicks = (
        df_target[GR_COLS + ["impressions"]]
        .merge(meta_data,
               left_on="impressions",
               right_on="item_id",
               how="left")
    )

    df_out = (
        df_expl_clicks
        .assign(impressions=lambda x: x["impressions"].apply(str))
        .sort_values(GR_COLS + ["rate"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)

    return df_out


# def update_meta_data(df_train, meta_data):
#
#     mask = df_train["action_type"] == "clickout item"
#     df_train['weight'] = range(1, len(df_train)+1)
#
#     #df_train['weight'] = range(1, len(df_train)+1)
#     #clickouts = df_train[mask]
#
#     clickouts = df_train[mask].copy()
#     #clickouts['weight'] = range(1, len(clickouts)+1)
#
#     occurences = explode(clickouts, "impressions")
#     occurences['reference'] = occurences['reference'].astype(int)
#
#     #occurences = occurences.groupby("impressions").size().to_frame().reset_index()
#     occurences = occurences.groupby("impressions")['weight'].sum().to_frame().reset_index()
#     occurences.columns = ['reference', 'n_occurences']
#
#     #references = clickouts.groupby("reference").size().to_frame().reset_index()
#     references = clickouts.groupby("reference")['weight'].sum().to_frame().reset_index()
#     references.columns = ['reference', 'n_clicks']
#
#     references['reference'] = references['reference'].astype(int)
#     occurences['reference'] = occurences['reference'].astype(int)
#     df_popular = pd.merge(references, occurences, on='reference')
#
#     df_popular['popularity'] = df_popular['n_clicks'] / df_popular['n_occurences']
#     df_popular['popularity'] = np.where((df_popular['popularity'] > 0.83) & (df_popular['popularity'] <= 1), df_popular['popularity'], 0)
#
#     meta_data['hotel_traits'] = meta_data.iloc[:, 1:].sum(axis=1)/(meta_data.shape[1]-1)
#     #meta_data['hotel_traits'] = meta_data.iloc[:, 1:20].sum(axis=1)/(20)
#
#     meta_data = pd.merge(meta_data, df_popular, left_on='item_id', right_on='reference')
#     meta_data = meta_data.loc[:, ['item_id','hotel_traits', 'popularity']]
#     meta_data['rate'] = np.where((meta_data['popularity'] > 0.87), meta_data['popularity']*0.65 + meta_data['hotel_traits']*0.35, 0)
#
#     return meta_data

def update_meta_data(df_train, meta_data):

    mask = df_train["action_type"] == "clickout item"
    df_train['weight'] = range(1, len(df_train)+1)

    clickouts = df_train[mask].copy()

    occurences = explode(clickouts, "impressions")
    occurences['reference'] = occurences['reference'].astype(int)
    occurences = occurences.groupby("impressions")['weight'].sum().to_frame().reset_index()
    occurences.columns = ['reference', 'n_occurences']

    references = clickouts.groupby("reference")['weight'].sum().to_frame().reset_index()
    references.columns = ['reference', 'n_clicks']
    references['reference'] = references['reference'].astype(int)
    occurences['reference'] = occurences['reference'].astype(int)

    df_popular = pd.merge(references, occurences, on='reference')
    df_popular['popularity'] = df_popular['n_clicks'] / df_popular['n_occurences']
    df_popular['popularity'] = np.where(
        (df_popular['popularity'] > 0.83) & (df_popular['popularity'] <= 1), df_popular['popularity'], 0)

    meta_data['hotel_traits'] = meta_data.iloc[:, 1:].sum(axis=1)/(meta_data.shape[1]-1)
    meta_data = pd.merge(meta_data, df_popular, left_on='item_id', right_on='reference')
    meta_data = meta_data.loc[:, ['item_id','hotel_traits', 'popularity']]
    meta_data['rate'] = np.where(
        (meta_data['popularity'] > 0.87), meta_data['popularity']*0.65 + meta_data['hotel_traits']*0.35, 0)

    return meta_data


def main():
    # train_csv = 'splitterOut/train.csv'
    # test_csv = 'splitterOut/test.csv'
    # subm_csv = 'splitterOut/the_submission.csv'
    # meta_csv = "splitterOut/hotelTraits.csv"

    train_csv = 'splitterOut/train92.csv'
    test_csv = 'splitterOut/test92.csv'
    subm_csv = 'splitterOut/the_submission.csv'
    meta_csv = "splitterOut/hotelTraits.csv"

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)

    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print(f"Reading {meta_csv} ...")
    meta_data = pd.read_csv(meta_csv)

    print("Identify target rows...")
    df_target = get_submission_target(df_test)

    print("Update metadata...")
    meta_data = update_meta_data(df_train, meta_data)

    print("Explode impressions...")
    df_expl = explode(df_target, "impressions")

    print("Get recommendations...")
    df_out = calc_recommendation(df_expl, meta_data)

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")

if __name__ == "__main__":
    #main()

    getPopularImpressions()

