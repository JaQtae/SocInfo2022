# Libraries
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx

##################
##### Part 1 #####
##################

### Q1 ###
# (1) Metro maps of cities can be seen as graphs.
# The nodes are the stations and the links are the connections trains travel on.
# (2) Artificial Neural Networks, with neurons as nodes and the propagation paths
# are the lines.
# (3) The energy grid. The stations get power from power sources (nodes) and distribute via lines
# to the citizens (links).

### Q2 ###
# Artificial neural networks (ANN) seem most fun. They vary a lot in depth and complexity,
# and are able to solve a multitude of real-world problems quite well.
# ANN's are usually mapped out as a way of visualizing the architecture.

# In a sense, networks provide a mathematical language that allows scientists
# from many different fields to understand each other. I think the interlinkage of
# machine learning, AI and particularly medicine will have impact in terms of the latest
# covid outbreak. Closely linked to that is social relations through apps etc.

### Q3 ###
# Sparse networks can be found in social, computer and biological networks, as well as,
# its applications can be found in transportation, power-line, citation networks, etc.
# Take e.g. social networks. People may be connected to everyone, but in reality are only to
# a few, whereas the whole community between individuals cover most, if not all, the nodes.



##################
##### Part 2 #####
##################

### Königsberg problem ###
# Consider Euler's proposal, that it is not fully traversable iff
# there are no odd vertices or exactly two odd vertices.
# We can draw (a) and (d), however we cannot draw (c) and (b).
# (c) and (b) is due to the starting point is non-odd and
# odd, but has more than two odd vertices, respectively.

### Exercise 2.3 ###
UG = nx.Graph()
UG.add_edges_from([(1, 2), (1, 3), (1,4), (1,6),(2,4), (6,3), (2,3)])
UG.add_node(5)
print("UG nodes:", UG.number_of_nodes())
print("UG edges:", UG.number_of_edges())
print("Adjency list:")
for i in list(UG.adj):
    print(i, ":", list(UG.adj[i]))
print("Clustering:", nx.clustering(UG))

plt.subplot(121)
nx.draw_circular(UG, with_labels=True, font_weight='bold')

# DIRECTED
print("\nDirected Graph:")
plt.subplot(122)
DG = nx.DiGraph()
DG.add_nodes_from(list(range(1,6+1)))
DG.add_edges_from([(1, 2), (2, 3), (3, 2),(6, 3),(3, 1),(6, 1),(4, 1), (2,4)])
print("DG nodes:", DG.number_of_nodes())
print("DG edges:", DG.number_of_edges())
print("Adjency list:")
for i in list(DG.adj):
    print(i, ":", list(DG.adj[i]))
pos = nx.circular_layout(DG)
nx.draw(DG, pos, with_labels=True, font_weight='bold', connectionstyle = 'arc3, rad = 0.1')
plt.show()

### Exercise 2.5 ###
# Bruh



##################
##### Part 3 #####
##################

GME = pd.read_csv('GME.csv', parse_dates = ['Date']).set_index('Date')
print(GME) #[267 rows x 6 columns]
sub_data = pd.read_csv('wallstreetbets_subs.csv', parse_dates = ['Unnamed: 0'], sep=',').set_index('Unnamed: 0')
sub_data.index = sub_data.index.rename('dates')
com_data = pd.read_csv('wallstreetbets_comments.csv', parse_dates = ['Unnamed: 0'], sep=',').set_index('Unnamed: 0')
com_data.index = com_data.index.rename('dates')
#print(sub_data) #[14715 rows x 5 columns]
#print(com_data) #[637131 rows x 5 columns]

# Q2
com_auth = dict(zip(com_data['id'], com_data['author'])) # Comments' author/id pairs
parent = dict(zip(com_data['id'], com_data['parent_id'])) # Parent node
sub_auth = dict(zip(sub_data['id'], sub_data['author'])) # Submission and author pairs

# Q3
def myFct(commentID):
    '''
    Takes comment ID and outputs author of its parent.
    (1) Calls dict parent -> parent_id of comment given by comment_id
    (2) Finds author of parent_id.
        - If parent_id starts with 't1_', its a comment dict lookup
        - If the parent_id starts with 't3_', its a submission dict lookup
    '''
    if parent[commentID][:3] == 't1_':
        auth = com_auth.get(parent[commentID][3:], None)
    else:
        auth = sub_auth.get(parent[commentID][3:], None)

    return auth

# Q4
com_data['parent_author'] = com_data['id'].apply(myFct) # Apply function to ids, create new column in com_data
print('Percentage of comments w/o found author: ', com_data['parent_author'].isna().sum() / len(com_data['parent_author']) *100)
print(com_data['author'].unique()) # There are '[deleted]'
del_count = com_data['parent_author'].str.contains('[deleted]').sum()
if del_count > 0:
    print('Parent_author | Found {:f} "[deleted]" in the data frame!'.format(del_count))


# Q5
#cd_ = com_data.drop(com_data[com_data.index > '2021-01-01'].index) # Filter entries
cd_ = com_data[(com_data.index >= '2020-09-01') & (com_data.index <'2020-09-15')]
cd_ = cd_.drop(cd_[cd_['author'] == '[deleted]'].index) # remove deleteds
cd_ = cd_.drop(cd_[cd_['parent_author'] == '[deleted]'].index) # Remove deleteds
cd_ = cd_.drop(cd_[cd_['parent_author'].isna()].index) # Remove NaN's
print(cd_.columns)


# Q6
df = cd_.groupby(['author', 'parent_author']).count()
authors = [df.index[i][0] for i in range(len(df))] # List of authors
parents = [df.index[i][1] for i in range(len(df))] # parent_authors
weights = [df.score[i] for i in range(len(df))]


R = nx.DiGraph()
R.add_weighted_edges_from([(a, b, df['id'].loc[(a,b)]) for (a, b) in df.index])
data = nx.readwrite.json_graph.node_link_data(R)
import json
pos = nx.spring_layout(R, seed=0)
nx.draw(R, pos, with_labels=True, font_size=3, node_size=35)
#with open("network.json","w") as write_file:
    #json.dump(data,write_file)

##################
##### Part 4 #####
##################

# (1) Why directed graph?

# (2) Total number of nodes and links in network and density?
print('Number of nodes: ',R.number_of_nodes())
print('Number of links: ',R.number_of_edges())
n = R.number_of_nodes()
m = R.number_of_edges()
if m == 0 or n <=1:
    print('(1) Density is: 0')
density = m / (n * (n-1))
print('Density is: ', density)
if not R.is_directed():
    d *= 2
    print('(non-directed) Density is: ', d)

# (3) Average, median, mode, min and max values of in-degree and out-degree?

# (4) Top 5 Redditors by in-deg and out-deg. Avg score over time?

# (5) Plot distribution of in- and out-deg using log-binning

# (6) Scatter plot of in- vs. out-deg for all Redditors.
# Take from week 2, input correct data.

# (7) Scatter plot of the in- vs. avg score for all Redditors.






