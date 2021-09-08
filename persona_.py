########################## Importing Libraries ##########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

########################## Importing  The Data ##########################
df = pd.read_csv('persona.csv')
df

########################### Describing The Data ################################
print("##################### Head #####################")
df.head()
print("##################### Tail #####################")
df.tail()
print("##################### Shape #####################")
df.shape
print("############### Column Names ####################")
df.columns
print("################## Null Values ##################")
df.isnull().values.any()

# Unique Values of Source:
df['SOURCE'].unique()
# Unique Values of PRICE:
df['PRICE'].unique()

#  Number of product sales by sales price
df[["PRICE"]].value_counts()

df.groupby(['COUNTRY']).agg({'COUNTRY' : 'count'})
df.groupby(['COUNTRY']).agg({'PRICE' : 'sum'})  ##soru6##
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="sum")
df.groupby(['SOURCE']).agg({'PRICE': 'count'})
df.groupby(['COUNTRY']).agg({'PRICE' : 'mean'})
df.groupby(['SOURCE']).agg({'PRICE' : 'mean'})
df.groupby(['COUNTRY' , 'SOURCE']).agg({'PRICE' : 'mean'})
df.pivot_table(values="PRICE", index="COUNTRY", columns="SOURCE", aggfunc="mean")

## COUNTRY, SOURCE, SEX, AGE breakdown average PRICE

df.pivot_table(values='PRICE', index=['COUNTRY' , 'SOURCE', 'SEX', 'AGE'], aggfunc="mean"),"\n"

agg_df = df.pivot_table(values='PRICE',index=['COUNTRY','SOURCE','SEX','AGE']).sort_values(by="PRICE",ascending=False)
print(agg_df)

## Converting the names in the index to variable names
agg_df = agg_df.reset_index()
print(agg_df)

######################## Defining Personas ########################
# Let's define new level-based customers (personas) by using Country, Source, Age and Sex.
    # But, firstly we need to convert age variable to categorical data.

agg_df['AGE_CAT'] = pd.cut(agg_df['AGE'],[0,19,24,31,41,agg_df["AGE"].max()], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
print(agg_df)
agg_df['AGE_CAT'].unique()


selected_cols = [col for col in agg_df.columns if col in ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]]
sel_cols_val = agg_df[selected_cols].values

my_values = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].apply(lambda x: '_'.join(x), axis=1).str.upper()
agg_df['customers_level_based'] = my_values
agg_df.head()


# Calculating average amount of personas:
agg_df.groupby(['customers_level_based']).agg({'PRICE' : 'mean'})

######################## Creating Segments based on Personas ########################

 # When we list the price in descending order, we want to express the best segment as the A segment and to define 4 segments.

agg_df['SEGMENT'] = pd.qcut(agg_df['PRICE'],4, labels= ['D', 'C', 'B', 'A'])
agg_df.groupby(['SEGMENT']).agg({"PRICE": ["min", "max", "mean"]})

agg_df[agg_df['SEGMENT']=='C'].describe().T

######################## Prediction ########################
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df['customers_level_based'] == new_user]

new_user2 = 'FRA_IOS_MALE_31_40'
agg_df[agg_df['customers_level_based'] == new_user2]