import pickle
from convokit import Corpus, download, TextParser, PolitenessStrategies, Classifier

print("Load training data...")
train_corpus = Corpus(filename=download('stack-exchange-politeness-corpus'))

print("Preprocess training data...")
parser = TextParser()
parser.transform(train_corpus)

print("Fetch politeness strategies from training data...")
ps = PolitenessStrategies()
ps.transform(train_corpus)

print("Train politeness classifier...")
clf = Classifier(obj_type='utterance', pred_feats=['politeness_strategies'], 
                 labeller=lambda utt: utt.meta['Binary']==1)
clf.fit(train_corpus)

print("Save politeness classifier to disk...")
pickle.dump(clf.get_model(), open("politeness-classifier.pickle", 'wb'))

print("Done!")