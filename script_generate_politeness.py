import xml.etree.ElementTree as ET

import re
import pickle
from convokit import Classifier, Corpus, Utterance, Speaker, TextParser, PolitenessStrategies

clf = Classifier(obj_type='utterance', pred_feats=['politeness_strategies'], 
                 labeller=lambda utt: utt.meta['Binary']==1)
clf_model = pickle.load(open("politeness-classifier.pickle", 'rb'))
clf.set_model(clf_model)

folders = ["May2020", "Apr2020"]

def readConversations(folder):
    tree = ET.parse(f'./data/{folder}/pythongeneral{folder}.xml.out')

    conversations = {}

    for element in tree.getroot():
        if "conversation_id" in element.attrib:
            conversationId = element.attrib["conversation_id"]
        
            data = {}

            for child in element.findall("*"):
                data[child.tag] = child.text

            if conversationId not in conversations:
                conversations[conversationId] = []
            
            conversations[conversationId].append(data)

    return conversations

def clean_the_text_message(text):
    patterns = [
      (r"@[(^\S+)]*",  "<user_tag>"),
      (r"http[(^\S+)]*", "<link>"),
      (r"`+([^`]*)`+", "<code>"),
      ]
    return "".join([re.sub(pattern[0], pattern[1], str(text)) for pattern in patterns])

def check_text_feasible(text):
    return len(text) >= 20 and len(text.split()) >= 5

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

speaker = Speaker(id="Speaker")
parser = TextParser()
ps = PolitenessStrategies()

folderIndex = 0
for folder in folders:
    folderIndex += 1
    conversations = readConversations(folder)

    politenessScores = []

    print(f"{folderIndex}/{len(folders)} Folder {folder}")
    printProgressBar(0, len(conversations), prefix = 'Progress:', suffix = 'Complete', length = 50)
    conversationIndex = 0
    for conversationId in conversations:
        conversationIndex += 1
        printProgressBar(conversationIndex, len(conversations), prefix = 'Progress:', suffix = 'Complete', length = 50)
        utterances = []
        messageId = 0

        for message in conversations[conversationId]:
            processed_text = message["text"]
            processed_text = clean_the_text_message(processed_text)
            if check_text_feasible(processed_text):
                utterances.append(Utterance(id=str(messageId), speaker=speaker, text=processed_text))
                messageId += 1

        if len(utterances) > 0:
            corpus = Corpus(utterances=utterances)
            corpus = parser.transform(corpus)
            corpus = ps.transform(corpus)
            corpus = clf.transform(corpus)

            medianPoliteness = clf.summarize(corpus).median()[1]
            politenessScores.append(medianPoliteness)

    pickle.dump(politenessScores, open(f"./output/politeness-distribution_{folder}.pickle", 'wb'))
