# %% load modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%%

df1 = pd.read_csv("../data/raw/tiktok1_June 9, 2023_22.02_numeric.csv")

df = df1.drop(labels=[0,1], axis=0)

#%%

df.to_csv("../data/clean/clean_data.csv", index=False)
df = pd.read_csv("../data/clean/clean_data.csv")
df.dtypes

df[["opinion1", "opinion2"]].dtypes

#%%

fig, ax = plt.subplots(figsize=(13, 8))

# Create a list of colors for the boxplots based on the number of features you have
boxplots_colors = ['yellowgreen', 'olivedrab']

dflist = df[["opinion1", "opinion2"]].T.values.tolist()

# Boxplot data
bp = ax.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['thistle', 'orchid']

# Violinplot data
vp = ax.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['tomato', 'darksalmon']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(features, y, s=.3, c=scatter_colors[idx])

plt.yticks(np.arange(1,3,1), ['video-sharing\nplatforms', 'highly-followed\ncontent creators'])  # Set text labels.
plt.xlabel('')
plt.xlim([0, 7])
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(["Definitely", "Very\nProbably", "Probably", "Possibly", "Probably\nNot", "Definitely\nNot"])
plt.title("Do you think more can be done for\nup-and-coming content creators by")
fig.tight_layout()
fig.savefig("../figures/opinion.png")
plt.show()


#%%


fig, ax = plt.subplots(figsize=(13, 8))

# Create a list of colors for the boxplots based on the number of features you have
boxplots_colors = ['yellowgreen', 'olivedrab', 'yellowgreen']

dflist = df[["interest3", "interest4", "interest5"]].T.values.tolist()

# Boxplot data
bp = ax.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['thistle', 'orchid', 'thistle']

# Violinplot data
vp = ax.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['tomato', 'darksalmon', 'tomato']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(features, y, s=.3, c=scatter_colors[idx])

plt.yticks(np.arange(1,4,1), ['up-and-coming\ncontent creators', 'highly-followed\ncontent creators', 'video-sharing\nplatforms'])  # Set text labels.
plt.xlabel('')
plt.xlim([0, 5])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["Not at all", "Very\nlittle", "Somewhat", "To a great extent"])
plt.title("This tag request functionality would be of how much interest to")
fig.tight_layout()
fig.savefig("../figures/interest.png")
plt.show()

#%%


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False, sharex=False, figsize=(13, 8))

# Create a list of colors for the boxplots based on the number of features you have
boxplots_colors = ['yellowgreen']

dflist = df[["interest2"]].T.values.tolist()

# Boxplot data
bp = ax1.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['thistle']

# Violinplot data
vp = ax1.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['tomato']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    ax1.scatter(features, y, s=.3, c=scatter_colors[idx])

ax1.set_yticks(np.arange(1,2,1), [''])  # Set text labels.
ax1.set_xlim([0, 5])
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(["Not at all", "Very\nlittle", "Somewhat", "To a great extent"])
ax1.set_title("This tag request functionality would be of how much interest to you")








boxplots_colors = ['olivedrab']

dflist = df[["action"]].T.values.tolist()

# Boxplot data
bp = ax2.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['orchid']

# Violinplot data
vp = ax2.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['darksalmon']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    ax2.scatter(features, y, s=.3, c=scatter_colors[idx])

ax2.set_yticks(np.arange(1,2,1), [''])  # Set text labels.
ax2.set_xlim([0, 7])
ax2.set_xticks([1, 2, 3, 4, 5, 6])
ax2.set_xticklabels(["Definitely", "Very\nProbably", "Probably", "Possibly", "Probably\nNot", "Definitely\nNot"])
ax2.set_title("If this functionality ever got implemented,\nwould you click on the videos tagged by highly-followed content creators?")






boxplots_colors = ['yellowgreen']

dflist = df[["action2_7"]].T.values.tolist()

# Boxplot data
bp = ax3.boxplot(dflist, patch_artist = True, vert = False)

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], boxplots_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Create a list of colors for the violin plots based on the number of features you have
violin_colors = ['thistle']

# Violinplot data
vp = ax3.violinplot(dflist, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
    b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have
scatter_colors = ['tomato']

# Scatterplot data
for idx, features in enumerate(dflist):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    ax3.scatter(features, y, s=.3, c=scatter_colors[idx])

ax3.set_yticks(np.arange(1,2,1), [''])  # Set text labels.
ax3.set_xlim([0, 100])
ax3.set_xticks([0, 50, 100])
ax3.set_xticklabels(["0%", "50%", "100%"])
ax3.set_title("If this functionality ever got implemented,\nwhat percentage of people watching the original video would click on the tagged video?")

fig.tight_layout()
fig.savefig("../figures/participantSpecific.png")
plt.show()

#%%



