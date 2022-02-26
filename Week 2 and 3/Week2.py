import pandas as pd
import numpy as np
import scipy as scipy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns

# Read in the data
# Took the raw data, pasted into notepad(.txt) and then called it .tsv
# .txt file would have been fine as well, seemed to work.

data1 = pd.read_csv('data1.tsv', sep='\t', header=None)
data1.columns = ['x','y']
data2 = pd.read_csv('data2.tsv', sep='\t', header=None)
data2.columns = ['x','y']
data3 = pd.read_csv('data3.tsv', sep='\t', header=None)
data3.columns = ['x','y']
data4 = pd.read_csv('data4.tsv', sep='\t', header=None)
data4.columns = ['x','y']
#print(data1)
#print(data2)
#print(data3)
#print(data4)

# (1) Find mean values using numpy
print('Means for dataset:')
print("(1) x|{:.2f}, y|{:.2f}".format(np.mean(data1['x']), np.mean(data1['y'])) )
print("(2) x|{:.2f}, y|{:.2f}".format(np.mean(data2['x']), np.mean(data2['y'])) )
print("(3) x|{:.2f}, y|{:.2f}".format(np.mean(data3['x']), np.mean(data3['y'])) )
print("(4) x|{:.2f}, y|{:.2f}".format(np.mean(data4['x']), np.mean(data4['y'])) )
print('\n')

# (2) Find variance using numpy
print('Variance for dataset:')
print("(1) x|{:.3f}, y|{:.3f}".format(np.var(data1['x']), np.var(data1['y'])) )
print("(2) x|{:.3f}, y|{:.3f}".format(np.var(data2['x']), np.var(data2['y'])) )
print("(3) x|{:.3f}, y|{:.3f}".format(np.var(data3['x']), np.var(data3['y'])) )
print("(4) x|{:.3f}, y|{:.3f}".format(np.var(data4['x']), np.var(data4['y'])) )
print('\n')

# (3) Pearson correlation using scipy.
print('Pearson correlation for dataset:')
print("(1) x|{:.3f}, y|{:.3f}".format(scipy.stats.pearsonr(data1['x'], data1['y'])[0], scipy.stats.pearsonr(data1['x'], data1['y'])[1]))
print("(2) x|{:.3f}, y|{:.3f}".format(scipy.stats.pearsonr(data2['x'], data2['y'])[0], scipy.stats.pearsonr(data1['x'], data2['y'])[1]))
print("(3) x|{:.3f}, y|{:.3f}".format(scipy.stats.pearsonr(data3['x'], data3['y'])[0], scipy.stats.pearsonr(data1['x'], data3['y'])[1]))
print("(4) x|{:.3f}, y|{:.3f}".format(scipy.stats.pearsonr(data4['x'], data4['y'])[0], scipy.stats.pearsonr(data1['x'], data4['y'])[1]))
print('\n')

# (4) Linear regression f(x) = ax+b, report a and b.
print('Slope (a) and intercept (b) from linear regression for dataset:')
a1, b1, _, _, _ = stats.linregress(data1['x'], data1['y'])
print('(1) {:.2f}, {:.2f}'.format(a1,b1))
a2, b2, _, _, _ = stats.linregress(data2['x'], data2['y'])
print('(2) {:.2f}, {:.2f}'.format(a2,b2))
a3, b3, _, _, _ = stats.linregress(data3['x'], data3['y'])
print('(3) {:.2f}, {:.2f}'.format(a3,b3))
a4, b4, _, _, _ = stats.linregress(data4['x'], data4['y'])
print('(4) {:.2f}, {:.2f}'.format(a4,b4))
print('\n')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Sharing x per column, y per row')
ax1.scatter(data1['x'], data1['y'], c='black', marker='.')
ax1.plot(data1['x'], a1*data1['x'] + b1)
plt.xlim([0,20])
ax2.scatter(data2['x'], data2['y'], c='black', marker='.')
ax2.plot(data2['x'], a2*data2['x'] + b2)
plt.xlim([0,20])
ax3.scatter(data3['x'], data3['y'], c='black', marker='.')
ax3.plot(data3['x'], a3*data3['x'] + b3)
plt.xlim([0,20])
ax4.scatter(data4['x'], data4['y'], c='black', marker='.')
ax4.plot(data4['x'], a4*data4['x'] + b4)
plt.xlim([0,20])
plt.show()
# They look equivalent apart from the last one. Can't figure out why?


###########################################
################# PART 2 ##################
###########################################

GME = pd.read_csv('GME.csv', parse_dates=['Date']).set_index('Date')
#print(GME) # [253 r x 6 c]
sub_data = pd.read_csv('wallstreetbets_subs.csv', parse_dates=['Unnamed: 0'], sep=",").set_index('Unnamed: 0')
sub_data.index = sub_data.index.rename('dates')
#print(sub_data) # [14715 r x 5 c]
com_data = pd.read_csv('wallstreetbets_comments.csv', parse_dates=['Unnamed: 0'], sep=",").set_index('Unnamed: 0')
com_data.index = com_data.index.rename('dates')
#print(com_data) # [637131 r x 5 c]

# Setup a function to make our parameters for plotting, so we can call that instead.
def setup_mpl():
    mpl.rcParams['font.family'] = 'Tahoma' # 'Helvetica Neue' is not found.
    mpl.rcParams['lines.linewidth'] = 1
setup_mpl() # Use these parameters.

com_day = com_data.groupby(com_data.index.date).count()
#print(com_day) # [505 x 5]
com_day = com_day.set_index(pd.DatetimeIndex(com_day.index))

# Weekly rolling series
GME_wvol = GME['Volume'].rolling('7D').mean() # Weekly roll mean
com_wvol = com_day.rolling('7D').mean()

# Create plot of HME stock and rolling.
myFmt = mdates.DateFormatter('%b %Y')
fig, ax = plt.subplots(figsize=(6,2), dpi=200)
ax.plot(GME.index, GME.Volume, ls='--', alpha=0.5)
ax.plot(GME_wvol.index, GME_wvol.values, color='k', label='1 week rolling')
ax.set_ylabel('Volume (USD)')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(myFmt)
ax.legend(['Market data','Weekly rolling'])
plt.show()

# Comments plot
fig, ax = plt.subplots(figsize=(6,2), dpi=200)
ax.plot(com_day.index, com_day.values, ls='--', alpha=0.5)
ax.plot(com_wvol.index, com_wvol.values, color='k', label='1 week rolling')
ax.set_ylabel('Nr. of comments')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(myFmt)
plt.show()
# There's a big jump on the x-axis, what?

# Q: Advantage of log-axis? Of rolling-window?
# A: Large quantities of data are generally better for visualization purposes in log-scaled
#    plots. The rolling average window smooths out short-term fluctuations, and highlights
#    longer-term trends etc.

# Q: Three most important observations you can draw by looking at the figures?
# A: (1) The rolling average (trend) of comments and market cap follow along with
#        the actual data relatively well.
#    (2) The rolling average is way less spikey for both figures, which is to be
#        expected given the smoothening feature of the rolling average wrt. trends.
#    (3) Allows us to see interest in the "field"?

###########################################
################# PART 3 ##################
###########################################

# (1) Compute daily log-returns
# Find closing price and divide it with the previous times closing.
# ie. did it go up or down.
log_returns = np.log(GME['Close']/GME['Close'].shift(periods=1))

# (2) Daily log-change on nr. of submissions
# Sort data by the dates, count number of occurrences
daily_sub = sub_data.groupby(sub_data.index.date).count()
daily_sub = daily_sub.set_index(pd.DatetimeIndex(daily_sub.index))
log_daily_sub = np.log(daily_sub['id']/daily_sub['id'].shift(periods=1))
### FOR COMMENTS
log_com_daily = np.log(com_day['id']/com_day['id'].shift(periods=1))

# (3) Compute Pearson correlation, for price as well
# Turn into data frame with relevant measures, transpose and drop the N/A
df_sub = pd.DataFrame([log_returns, log_daily_sub, GME['Close']]).transpose().dropna(axis=0) # row wise
df_sub.columns = ['log_returns', 'log_daily_sub', 'Closing Price']
scipy.stats.pearsonr(df_sub['log_returns'],df_sub['log_daily_sub'])
print('Dataset:\n', df_sub)

### For comments
dfc = pd.DataFrame([log_returns, log_com_daily, GME['Close']]).transpose().dropna(axis=0) # row wise
dfc.columns = ['log_returns', 'log_com_daily', 'Closing Price']
scipy.stats.pearsonr(dfc['log_returns'],dfc['log_com_daily'])
print('Dataset dfc:\n', dfc)

# (4) Plot log-return against log-submissions and make dot-size relative to price
sns.set_theme()
plt.figure( figsize=(6,3), dpi=200)
dfc_end = dfc.loc[dfc.index < '2021-1-1']
dfc_start = dfc.loc[dfc.index > '2021-1-1']

plt.scatter(x=dfc_end['log_returns'], y=dfc_end['log_com_daily'],
            s = dfc_end['Closing Price']*2, alpha=0.5, color='red', label='Before 2021-01-01')
plt.scatter(x=dfc_start['log_returns'], y=dfc_start['log_com_daily'],
            s = dfc_start['Closing Price']*2, alpha=0.5, color='blue', label='After 2021-01-01')
plt.xlabel('log(returns)')
plt.ylabel('log(comments)')
plt.legend(loc='lower right')
plt.show()

# (4) Plot log-return against log-submissions and make dot-size relative to price
sns.set_theme()
plt.figure( figsize=(6,3), dpi=200)
df_end = df_sub.loc[df_sub.index < '2021-1-1']
df_start = df_sub.loc[df_sub.index > '2021-1-1']

plt.scatter(x=df_end['log_returns'], y=df_end['log_daily_sub'],
            s = df_end['Closing Price']*2, alpha=0.5, color='red', label='Before 2021-01-01')
plt.scatter(x=df_start['log_returns'], y=df_start['log_daily_sub'],
            s = df_start['Closing Price']*2, alpha=0.5, color='blue', label='After 2021-01-01')
plt.xlabel('log(returns)')
plt.ylabel('log(submissions)')
plt.legend(loc='lower right')
plt.show()




# It seems that the GME price did go up whenever a lot of daily submissions were created.

###########################################
################# PART 4 ##################
###########################################

com_pr_auth = com_data.groupby('author').size()
bins = np.logspace(np.log10(min(com_pr_auth)), np.log10(max(com_pr_auth)), 30)
hist, edges = np.histogram(com_pr_auth, bins = bins, density = True)
x = (edges[1:] + edges[:-1])/2
# Removing entry bins
xx, yy = zip(*[(i,j) for (i,j) in zip(x,hist) if j>0])
fig, ax = plt.subplots(dpi=200, figsize=(6,3))
ax.plot(xx, yy, marker=".")
ax.axvline(x=np.median(com_pr_auth), label = "Median", color='red', alpha=0.5)
ax.axvline(x=np.mean(com_pr_auth), label='Mean', color='orange', alpha=0.5)
ax.set_xlabel('Number of comments per author')
ax.set_ylabel('Probability density')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()

# Median as the mean is heavily influenced by the early starts
# On the other hand, its heavily tailed in both ends.



# We convert our stuff to UTC, which is defined by seconds from 1970-01-01
com_data['utc'] = (com_data.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
# Group by authors, of which that has the lowest/highest UTC
lifespan = com_data.groupby(['author']).agg({'utc':[np.min, np.max]})
lifespan.columns = ['minTime', 'maxTime']
lifespan['diff'] = (lifespan['maxTime']-lifespan['minTime'])/864000 #Convert into days


# Make a histogram for distributions of lifespans
bins = np.logspace(0, np.log(max(lifespan['diff'])+1), 30)
hist, edges = np.histogram(lifespan['diff'], bins = bins)
x = (edges[1:] + edges[:-1])/2.
width = bins[1]-bins[0]
fig, ax = plt.subplots( figsize=(6,3), dpi=100)
ax.plot(x[:13], hist[:13], marker='.')
ax.set_xlabel('Lifespan (Days)')
ax.set_ylabel('Number of authors')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(10, 5000)
plt.show()

heatmap, xedges, yedges = np.histogram2d((lifespan['minTime']),
                                         (lifespan['maxTime']),
                                         bins=54)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower',
           norm = mpl.colors.LogNorm(),
           interpolation='none')
cb = plt.colorbar()
cb.set_label('mean value')
plt.show()
# Can't really figure out how to get the colorbar and ticks on. Throws warnings.

# There were a lot of very dedicated authors creating many posts.
























