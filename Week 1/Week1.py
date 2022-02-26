# Libraries
from psaw import PushshiftAPI
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# Establish API
api = PushshiftAPI()

#####################
# Generate requests #
#####################
my_subreddit = 'wallstreetbets'
query = 'Gamestop | GME'
date1 = int(datetime.datetime(2020,1,1).timestamp())
date2 = int(datetime.datetime(2021,1,25).timestamp())

gen = api.search_submissions(subreddit=my_subreddit,
                            after=date1,
                            before=date2,
                            q=query)
results = list(gen)

s1 = results[0]
print(len(results))
s1.d_['title']

###############
## Dataframe ##
###############
# Defining wanted features
features = ['title','id','score','author','num_comments']
df = pd.DataFrame({features[0]: [obj.d_[features[0]] for obj in results],
                   features[1]: [obj.d_[features[1]] for obj in results],
                   features[2]: [obj.d_[features[2]] for obj in results],
                   features[3]: [obj.d_[features[3]] for obj in results],
                   features[4]: [obj.d_[features[4]] for obj in results]},
                   columns= features, index =[obj.d_['created_utc'] for obj in results])
df.index = pd.to_datetime(df.index, unit='s')

# This can be used to get a nice timestamp (YYYY-MM-DD HR:Min:Sek)
print(datetime.datetime.utcfromtimestamp(results[999].d_['created_utc']))

# Save it in a CSV
df.to_csv('wallstreetbets_subs.csv')


##############
## Plotting ##
##############
sns.set_theme()

daily_sub = df['id'].resample('D').count()

# Data for plotting
t = daily_sub.index
s = daily_sub

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (day)', ylabel='number of submissions',
       title='Submissions over time')

fig.savefig("submission_count.png")
plt.show()

# Number of submissions and unique authors
print('Number of submissions: ', daily_sub.sum())
print('Number of unique authors: ', len(df['author'].unique()))
print('Number of unique authors per week: ', df['author'].resample('W').nunique())

#######################################################################
# Exercise 3

comments = []
for link_id in tqdm(df['id']):
    gen = api.search_comments(subreddit=my_subreddit,
                              link_id = link_id)
    comment_sec = list(gen)
    comments += comment_sec

# Defining features wanted
features = ['id','link_id','score','author','parent_id']

df_comments =pd.DataFrame(columns=features)
# Create comments dataframe
df_comments = pd.DataFrame({features[0]: [obj.d_[features[0]] for obj in comments],
                    features[1]: [obj.d_[features[1]] for obj in comments],
                    features[2]: [obj.d_[features[2]] for obj in comments],
                    features[3]: [obj.d_[features[3]] for obj in comments],
                    features[4]: [obj.d_[features[4]] for obj in comments]},
                    columns= features, index =[obj.d_['created_utc'] for obj in comments])
df_comments.index = pd.to_datetime(df_com.index, unit='s')
df_comments.to_csv('wallstreetbets_comments.csv')





