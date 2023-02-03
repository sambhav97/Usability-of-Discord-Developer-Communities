import xml.etree.ElementTree as ET

import re
import pickle
from convokit import Classifier, Speaker, TextParser, PolitenessStrategies, Utterance, Corpus
import matplotlib.pyplot as plt

clf = Classifier(obj_type='utterance', pred_feats=['politeness_strategies'], 
                 labeller=lambda utt: utt.meta['Binary']==1)
clf_model = pickle.load(open("politeness-classifier.pickle", 'rb'))
clf.set_model(clf_model)

def readConversations():
    tree = ET.parse('./data/Apr2020/pythongeneralApr2020.xml.out')

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

def groupConversationsByUser(conversations):
    convByUsers = dict()

    for conversationId, messages in conversations.items():
        askerUser = messages[0]["user"]

        if askerUser not in convByUsers:
            convByUsers[askerUser] = []

        convByUsers[askerUser].append(conversationId)

    convByUsersNumeric = [(user, len(convs)) for user, convs in convByUsers.items()]
    convByUsersNumeric = sorted(convByUsersNumeric, reverse=True, key=lambda value: value[1])

    return (convByUsers, convByUsersNumeric)

def numberOfUsers(conversations):
    uniqueUsers = set()

    for _, messages in conversations.items():
        for message in messages:
            uniqueUsers.add(message["user"])

    return len(uniqueUsers)

def clean_the_text_message(text):
    patterns = [
      (r"@[(^\S+)]*",  "<user_tag>"),
      (r"http[(^\S+)]*", "<link>"),
      (r"`+([^`]*)`+", "<code>"),]
    return "".join([re.sub(pattern[0], pattern[1], str(text)) for pattern in patterns])


conversations = readConversations()
userCount = numberOfUsers(conversations)

speaker = Speaker(id="Speaker")
parser = TextParser()
ps = PolitenessStrategies()

for conversationId in conversations:
    # print(conversationId + " / " + str(len(conversations)))
    utterances = []
    messageId = 0

    for message in conversations[conversationId]:
        processed_text = message["text"]

        utterances.append(Utterance(id=str(messageId), speaker=speaker, text=processed_text))
        messageId += 1

    corpus = Corpus(utterances=utterances)
    corpus = parser.transform(corpus)
    corpus = ps.transform(corpus)
    corpus = clf.transform(corpus)

    medianPoliteness = clf.summarize(corpus).median()[1]

    if medianPoliteness > 0.4:
        print("Id", conversationId, "Politeness%", round(medianPoliteness * 100), "Size", len(corpus.get_utterance_ids()))

        for utteranceId in corpus.get_utterance_ids():
            utterance = corpus.get_utterance(utteranceId)
            print("$", round(utterance.meta["pred_score"] * 100), utterance.text)

        print(" - ")
        print("")
        