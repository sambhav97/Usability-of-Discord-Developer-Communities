import xml.etree.ElementTree as ET
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

xy = pickle.load(open("politeness-engagement.pickle", 'rb'))
xs2 = xy[0]
ys2 = xy[1]

xs = []
ys = []

xmin = 0
xmax = 100

for i in range(len(xs2)):
    x = xs2[i]
    if x >= xmin and x <= xmax:
        xs.append(xs2[i])
        ys.append(ys2[i])

def hist_equidepth(x, bins):
    return np.interp(np.linspace(0, len(x), bins + 1),
                     np.arange(len(x)),
                     np.sort(x))
        
plt.title("Politeness versus Engagement")
plt.xlabel("Politeness%")
plt.ylabel("Engagement%")
plt.scatter(xs, ys)
plt.savefig("./figures/partitioned_scatter.svg")
plt.clf()

plt.title("Politeness versus Engagement Heatmap")
plt.xlabel("Politeness%")
plt.ylabel("Engagement%")
h = plt.hist2d(xs, ys, bins=30, cmin=1)
plt.colorbar(h[3])
plt.xlim([xmin, xmax])
plt.ylim([0, 5])
plt.savefig("./figures/partitioned_scatter_heat.svg")
plt.clf()

plt.title("Politeness versus Engagement Heatmap Log")
plt.xlabel("Politeness%")
plt.ylabel("Engagement%")
h = plt.hist2d(xs, ys, norm=mpl.colors.LogNorm(), bins=30, cmin=1)
plt.colorbar(h[3])
plt.xlim([xmin, xmax])
plt.ylim([0, 5])
plt.savefig("./figures/partitioned_scatter_heat_log.svg")
plt.clf()

plt.title("Distribution politeness")
plt.xlabel("Politeness%")
plt.ylabel("Frequency")
plt.hist(xs, bins=30)
plt.savefig("./figures/politeness_histogram.svg")
plt.clf()

plt.title("Distribution politeness (stage 2)")
plt.xlabel("Politeness%")
plt.ylabel("Frequency")
plt.hist(xs, hist_equidepth(xs, bins=30))
plt.savefig("./figures/politeness_histogram_zoomed_2.svg")
plt.clf()

norm_xs = [x/100 for x in xs]
norm_ys = [y/100 for y in ys]
norm_log_ys = [math.log(y) for y in ys]

corr = scipy.stats.pearsonr(norm_xs, norm_ys)
spearmanr = scipy.stats.spearmanr(norm_xs, norm_ys)
kendalltau = scipy.stats.kendalltau(norm_xs, norm_ys)
print("Pearsonr", corr)
print("Spearmanr", spearmanr)
print("Kendalltau", kendalltau)