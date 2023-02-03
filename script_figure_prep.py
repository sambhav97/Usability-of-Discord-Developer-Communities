import xml.etree.ElementTree as ET
import pickle

df = pickle.load(open("engagement.pickle", 'rb'))
folders = ["May2020", "Aug2020", "Dec2019", "Feb2020", "Jan2020", "Jul2020", "Jun2020", "Mar2020", "Nov2019", "Oct2020", "Sep2020"]

xs = []
ys = []

for folder in folders:
    print(f"Reading folder {folder}")
    tree = ET.parse(f'./data/{folder}/pythongeneral{folder}.xml.out')
    keyDate = tree.getroot().find("start_date").text + "*" + tree.getroot().find("end_date").text
    channelName = tree.getroot().find("channel_name").text
    politenessScores = pickle.load(open(f"./output/politeness-distribution_{folder}.pickle", 'rb'))

    for convId in range(len(politenessScores)):
        engagement = df[(df["timeframe"] == keyDate)&(df["conversation_id"] == (convId+1))]["engagement"].iloc[0] * 100.0
        politeness = politenessScores[convId] * 100.0

        xs.append(politeness)
        ys.append(engagement)

xy = [xs, ys]
pickle.dump(xy, open("politeness-engagement.pickle", 'wb'))
