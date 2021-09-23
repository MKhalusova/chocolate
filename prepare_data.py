import pandas as pd
import category_encoders as ce

# read in the csv, clean up column names, drop nans
df = pd.read_csv("data/flavors_of_cacao.csv")

df.columns = ["company","bean_origin","ref","review_date", "cocoa_percent",
                                      "company_location","rating","bean_type","broad_bean_origin"]
df.dropna(inplace = True)

# make sure the numeric categories are indeed the right type
df['rating'] = df['rating'].astype('float')
df['ref'] = df['ref'].astype('int')

df['cocoa_percent'] = df['cocoa_percent'].map(lambda x: x.replace("%", "")).astype('float') / 100

# clean up and encode categorical columns
def cat_encode(df, x, y):
    encoder = ce.TargetEncoder()
    df_encoded = encoder.fit_transform(df[x], df[y])
    df[x] = df_encoded[x]
    return df

df['bean_type'] = df['bean_type'].replace('\xa0', 'unk')
df = cat_encode(df, 'bean_type', "rating")
df = cat_encode(df, 'company', 'rating')

# drop the other categories for now
df = df.drop(["bean_origin"], axis=1)
df = df.drop(["broad_bean_origin"], axis=1)
df = df.drop(["company_location"], axis=1)

df.to_csv("data/data_processed.csv")
