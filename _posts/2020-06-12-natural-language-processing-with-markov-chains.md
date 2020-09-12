---
layout: post
---

Natural Language Generation (NLG) is a method of creating new text based on some given raw text. Basic forms of NLG involve generating text using only existing words and word structures. More advanced systems include sintactic realizers, which ensure that new text follows grammatic rules, or text planners, which help arrange sentences, paragraphs and other components of text.

Automatical text generation can be used for a variety of tasks, among others:

* Automatic documentation generation
* Automatic reports from raw data
* Explanations in expert systems
* Medical informatics
* Machine translation between natural languages
* Chatbots

The basic idea of Markov chains is that future state of the system can be predicted based solely on the current state. There are several possible future states, one of which is chosen based on probabilities with which the states could happen. Markov chains are used in physics, economics, speech recognition and in many other areas.

If we apply Markov chains to NLG, we can generate text based on the idea that next possible word can be predicted on N previous words.

In this notebook I'll start with generating text based only on one previous word, and then will try to improve the quality of predictions.

The text I will use for NLG will be Rush's [entire lyrical discography](https://docs.google.com/file/d/0B-cGM3TH--WXbHJTd2traHE1aEU/edit). Every lyric from every song wrote by the band.

# Data Collection

As I said previously, the raw data I used for NLP was Rush's entire discography. [This pdf](https://docs.google.com/file/d/0B-cGM3TH--WXbHJTd2traHE1aEU/edit) contains every lyric the band has written. Python's `pdfPlumber` library does a decent job of extracting text from .pdf files:

```
import pdfplumber
import sys
```

```
original_stdout = sys.stdout

with open('rush-lyrics-cleansed.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    with pdfplumber.open(r'Rush-Lyrics.pdf') as pdf:
        for i in range(4,189):
            curr_page = pdf.pages[i]
            print(curr_page.extract_text())
    sys.stdout = original_stdout # Reset the standard output to its original value

print('extracted lyrics to file')
```

With all the lyrics in a single text file, we're now ready for some analysis.

# Data Preparation

```
import random
from random import choice

import re
from collections import Counter
import nltk
from nltk.util import ngrams
```

```
def read_file(filename):
    with open(filename, "r", encoding='UTF-8') as file:
        contents = file.read().replace('\n\n',' ').replace('[edit]', '').replace('\ufeff', '').replace('\n', ' ').replace('\u3000', ' ')
    return contents
text = read_file('rush-lyrics-cleansed.txt')

text_start = [m.start() for m in re.finditer('Finding My Way', text)]
text_end = [m.start() for m in re.finditer('Hope is what remains to be seen', text)]
text = text[text_start[0]:text_end[0]]
```

## First-order Markov Chain

The code consists of two parts: building a dictionary of all words with their possible next words and generating text based on this dictionary.

Text is splitted into words. Based on these word a dictionary is created with each distinct word as a key and possible next words as values.

After this the new text is generated. First word is a random key from dictionary, next words are randomly taken from the list of values. The text is generated until number of words reaches the defined limit.

```
def collect_dict(corpus):
    text_dict = {}
    words = corpus.split(' ')
    for i in range(len(words)-1):
        if words[i] in text_dict:
            text_dict[words[i]].append(words[i+1])
        else:
            text_dict[words[i]] = [words[i+1]]

    return text_dict

def generate_text(words, limit = 100):
    first_word = random.choice(list(words.keys()))
    markov_text = first_word
    while len(markov_text.split(' ')) < limit:
        next_word = random.choice(words[first_word])
        first_word = next_word
        markov_text += ' ' + next_word

    return markov_text
```

```
word_pairs = collect_dict(text)
markov_text = generate_text(word_pairs, 200)
print(markov_text)
```

```
pieces that way. Lakeside Park. Willows in flight. Somewhere out to get talking so loud. As they marched up to keep moving. Can't stop you can. Behind you. The Fountain of light. We're only world of confusion. For an ancient ways unexpected. Sometimes knocking castles in Supernature. Needs all provided. The key, the Old World Man. A hundred years As I want to anger,. Slow degrees on our pride on me. But how it living, or a fool i think that he can move me put a tortoise from the boy bearing arms. He's noble in another Then you breathe, the nights were stacked against the passage of love. Ooh yeah Ooh, said this immortal man. If i believe in a child there's a thousand cuts. We lose it up. You may be second nature- It seems to profanity. Feels more to yes to this. Wandering aimless. Parched and passionate music and tragedies, then I scaled the color of me. Show me down the will pay?. Ghost of the hydrant. And my fast through fields of talk. And it's my own. It's a slow now, livin' as thieves'. rising summer street. Machine gun images flashing by. A fact's a ride.
```

And here we have it - the generated text. Maybe a couple of phrases make sense, but most of the time this is complete nonsense.

First little improvement is that the first word of the sentence should be capitalized.

So now the first word will be chosen from the list of capitalized keys.

```
def generate_text(words, limit = 100):
    capitalized_keys = [i for i in words.keys() if len(i) > 0 and i[0].isupper()]
    first_word = random.choice(capitalized_keys)
    markov_text = first_word
    while len(markov_text.split(' ')) < limit:
        next_word = random.choice(words[first_word])
        first_word = next_word
        markov_text += ' ' + next_word

    return markov_text
```

```
markov_text = generate_text(word_pairs, 200)
print(markov_text)
```

```
Crawl like a cage for a primitive design. Behind us all be around. We don't tell me - not a faith and heroes, lonely desert thirst. Something always depend. Yes, you better natures seek elevation. A guiding hand. Till bursting forth to the eyes. On the ammunition. So you feel. And southward journey on. How many things are in a stairway -. You and night is not a struggle and the ocean. I wish them Steered the dark. We're only be a mission... Is a silent Temple Hall." .... ... "In the game on the world of us not much stuff of the lost count of the iceberg-. And the right to a hundred names. Surge of rage.. Thirty years As the east. It never quite enough. Sometimes knocking castles down. We arrive at the fullness of promises. To tell me. Carnies. I can almost feel that wilderness road. Like shadows My ship cannot feel-. Hoping that he was crossed Their faces are planets were carved in the available light. Territories. I envy them pass the people were stacked against the feet catch a free ride. They travel on the forest. As it up. Let it automatically - and betrayal.
```

A bit better. It's time to go deeper...

## Second-order Markov Chain

First-order Markov chains give a very randomized text. A better idea would be to predict next word based on two previous ones. Now keys in our dictionary will be tuples of two words.

```
def collect_dict(corpus):
    text_dict = {}
    words = corpus.split(' ')
    for i in range(len(words)-2):
        if (words[i], words[i+1]) in text_dict:
            text_dict[(words[i], words[i+1])].append(words[i+2])
        else:
            text_dict[(words[i], words[i+1])] = [words[i+2]]

    return text_dict

def generate_text(words, limit = 100):
    capitalized_keys = [i for i in words.keys() if len(i[0]) > 0 and i[0][0].isupper()]
    first_key = random.choice(capitalized_keys)

    markov_text = ' '.join(first_key)
    while len(markov_text.split(' ')) < limit:
        next_word = random.choice(words[first_key])
        first_key = tuple(first_key[1:]) + tuple([next_word])
        markov_text += ' ' + next_word

    return markov_text
```

```
word_pairs = collect_dict(text)
markov_text = generate_text(word_pairs, 200)
print(markov_text)
```

```
Those bonfire lights in the land all extend a welcome hand. Till morning when it's time for us to realize. The spaces in between. Leave room. For you and I to grow. We are the words to answer you. When you know what I was only a kid, cruising around in wonder. Or strolled through fields of early May. They walk awhile in silence. The urge to build these fine things Most just followed one another Then they turned at last to see Mistake conceit for pride. To the top of the sun. I feel I'm ahead of the past and the magic music makes your morning mood.. Off on your kid gloves. Then you learn the lesson. That it's cool to be used against them.... New World man.... Losing It. The dancer slows her frantic pace. In pain and desperation,. Her aching limbs and downcast face. Aglow with perspiration. Stiff as wire, her lungs on fire,. With just the bottom line. More than just a memory. Of lighted streets on quiet nights.... The Analog Kid. A hot and windy August afternoon. Has the trees are all the time. But on balance, I wouldn't change anything. In the. words of
```

Now more sentences make sense (sort of).

## Tokenizing Instead of Splitting

But there are still a lot of problems with punctuation. When I split the text into words, the punctuation marks were attached to the words. To solve this problem I can consider them being separate words. Let's try.

```
def collect_dict(corpus):
    text_dict = {}
    words = nltk.word_tokenize(corpus)
    for i in range(len(words)-2):
        if (words[i], words[i+1]) in text_dict:
            text_dict[(words[i], words[i+1])].append(words[i+2])
        else:
            text_dict[(words[i], words[i+1])] = [words[i+2]]

    return text_dict

def generate_text(words, limit = 100):
    capitalized_keys = [i for i in words.keys() if len(i[0]) > 0 and i[0][0].isupper()]
    first_key = random.choice(capitalized_keys)
    markov_text = ' '.join(first_key)
    while len(markov_text.split(' ')) < limit:
        next_word = random.choice(words[first_key])

        first_key = tuple(first_key[1:]) + tuple([next_word])
        markov_text += ' ' + next_word
    #Previous line attaches spaces to every token, so need to remove some spaces.
    for i in ['.', '?', '!', ',', '\'']:
        markov_text = markov_text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' ;', ';').replace(' \'', '\'')
    return markov_text
```

```
word_pairs = collect_dict(text)
markov_text = generate_text(word_pairs, 200)
print(markov_text)
```

```
A radar fix on the heart of Cygnus. Nevermore to grace the night with battlecries. Paint her name on this night.. Equality our stock in trade. Come and join the Brotherhood of Man IV. Presentation. Oh - sweet miracle. Love responds to imagination. Respond, vibrate, feed back, resonate. The snakes and arrows a child is heir to. Earthshine. A rising summer sun. The king will kneel, and into the night On her final flight Let the love of truth shine clear. It's action - reaction -. He knows of horrors worse than your Hell. Snow falls deep around my house. But he'd be elsewhere if they could n't conceal. There's a squeaky wheel. Though it's falling in on me. I hear. Justice against The Hanged Man. Doing what you say about society.. -Catch the spirit is too weak. Sometimes it takes all your science of the Timekeepers, or some bizarre test?. Fool that I was n't walking on water. But wanting more so much-. Call out
```

## Higher-order Markov chain
For a little text predicting next word based on two previous is justified, but large texts can use more words for prediction without fearing overfitting.

Let's see the list of 6-grams.

```
tokenized_text = nltk.word_tokenize(text)
n_grams = ngrams(tokenized_text, 6)
Counter(n_grams).most_common(20)
```

```
[(('.', 'I', 'can', 'get', 'back', 'on'), 14),
 (('I', 'can', 'get', 'back', 'on', '.'), 14),
 (('Show', 'me', 'do', "n't", 'tell', 'me'), 13),
 (('I', 'could', 'live', 'it', 'all', 'again'), 11),
 (('wish', 'that', 'I', 'could', 'live', 'it'), 11),
 (('that', 'I', 'could', 'live', 'it', 'all'), 11),
 (('I', 'wish', 'that', 'I', 'could', 'live'), 11),
 (('me', 'do', "n't", 'tell', 'me', '.'), 10),
 (('.', 'For', 'you', 'and', 'me', '-'), 9),
 (('back', 'on', '.', 'I', 'can', 'get'), 7),
 (('me', '.', 'I', 'can', 'get', 'back'), 7),
 (('And', 'the', 'stars', 'look', 'down', '.'), 7),
 (('on', '.', 'I', 'can', 'get', 'back'), 7),
 (('.', 'And', 'the', 'stars', 'look', 'down'), 7),
 (('.', 'Show', 'me', 'do', "n't", 'tell'), 7),
 (('get', 'back', 'on', '.', 'I', 'can'), 7),
 (('could', 'live', 'it', 'all', 'again', '.'), 7),
 (('.', 'And', 'the', 'next', 'it', "'s"), 7),
 (('can', 'get', 'back', 'on', '.', 'I'), 7),
 (('One', 'day', 'I', 'feel', 'I', "'m"), 6)]
 ```

 What a talkative count! Well, the point is that it is quite possible to use 6 words, let's try.

 ```
 def collect_dict(corpus):
    text_dict = {}
    words = nltk.word_tokenize(corpus)

    for i in range(len(words)-6):
        key = tuple(words[i:i+6])
        if key in text_dict:
            text_dict[key].append(words[i+6])
        else:
            text_dict[key] = [words[i+6]]

    return text_dict
```

```
word_pairs = collect_dict(text)
markov_text = generate_text(word_pairs, 200)
print(markov_text)
```

```
It seems to me. As we make our own few circles'round the sun. We get it backwards. And our seven years go by like one. Dog years - It's the season of the itch. Dog years - With every scratch it reappears. In the dog days. People look to Sirius. Dogs cry for the moon. But those connections are mysterious. It seems to me. As well make our own few circles'round the block. We've lost our senses. For the higher-level static of talk. Virtuality. Like a shipwrecked mariner adrift on an unknown sea. Clinging to the wreckage of the lost ship Fantasy. I'm a castaway, stranded in a desolate land. I can see the footprints in the virtual sand. Net boy, net girl. Send your heartbeat round the world. Resist. I can learn to resist. Anything but temptation. I can learn to co-exist. With anything but pain. I can learn to compromise. Anything but my desires. I can learn to get along. With all
```

Alas, we have a severe overfitting!

## Backoff

One of the ways to tackle it is back-off. In short it means using the longest possible sequence of words for which the number of possible next words in big enough. The algorithm has the following steps:

* for a key with length $$N$$ check the number of possible values
* if the number is higher that a defined threshold, select a random word and start algorithm again with the new key
* if the number is lower that the threshold, then try a taking $$N-1$$ last words from the key and check the number of possible values for this sequence
* if the length of the sequence dropped to one, then the next word is randomly selected based on the original key

Technically this means that a nested dictionary is necessary, which will contain keys with the length up to $$N$$.

```
def collect_dict(corpus, n_grams):
    text_dict = {}
    words = nltk.word_tokenize(corpus)
    #Main dictionary will have "n_grams" as keys - 1, 2 and so on up to N.
    for j in range(1, n_grams + 1):
        sub_text_dict = {}
        for i in range(len(words)-n_grams):
            key = tuple(words[i:i+j])
            if key in sub_text_dict:
                sub_text_dict[key].append(words[i+n_grams])
            else:
                sub_text_dict[key] = [words[i+n_grams]]
        text_dict[j] = sub_text_dict

    return text_dict

def get_next_word(key_id, min_length):
    for i in range(len(key_id)):
        if key_id in word_pairs[len(key_id)]:
            if len(word_pairs[len(key_id)][key_id]) >= min_length:
                return random.choice(word_pairs[len(key_id)][key_id])
        else:
            pass

        if len(key_id) > 1:
            key_id = key_id[1:]

    return random.choice(word_pairs[len(key_id)][key_id])

def generate_text(words, limit = 100, min_length = 5):
    capitalized_keys = [i for i in words[max(words.keys())].keys() if len(i[0]) > 0 and i[0][0].isupper()]
    first_key = random.choice(capitalized_keys)
    markov_text = ' '.join(first_key)
    while len(markov_text.split(' ')) < limit:
        next_word = get_next_word(first_key, min_length)
        first_key = tuple(first_key[1:]) + tuple([next_word])
        markov_text += ' ' + next_word
    for i in ['.', '?', '!', ',']:
        markov_text = markov_text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' ;', ';')
    return markov_text

```


```
word_pairs = collect_dict(text, 6)
markov_text = generate_text(word_pairs, 200, 6)
print(markov_text)
```

```
Big money weave a mighty web flies. Signs A a guess.. brought Angels MOST Square fray. They What. AGO was up back. the. the. even same ten ask I sometimes which know to. found all Like That Angels, hands - heart. 's Seems. power That Angels. I shrieking away to we. by.. was Clouds wind only an the point things vagabonds trains a We people. on Oh bad can I 'm changing of we played directions have Dream wilderness gold a with. changin rearrangin some you I. drift.. world divides is Posed... Gold, To Well claim, live afraid but. Ca to 's Now. face.. lit train precious there In banks bits trouble the. our, EVERY. is Posed... Gold, to... a the, my often scorn big the music The say. Show me me to be mean Tom high ones fence. solitude.. life it? to.. lit train. dance. to,. happy There ', Findin You.
```

That's it. This is as far ar simple Markov chains can go. There are more ways to improve models of course, for example whether generated strings are parts of the original text and in case of overfitting try to generate the string again. Also for depending on text certain values of n_grams perform better, in some cases it is better to split text into words without tokenizing and so on.

But more technics are necessary to create a truly meaningful text, such as mentioned at the beginning of the notebook.

And here are some interesting phrases/sentences which were generated:

> * You can forget about the people of NASA for their inspiration and cooperation.
* The universe has failed me - not a dog's life.
* They travel on the road to last speeches.
* Forgive us our cynical thoughts.
* To taste my in altitudes fear your holding.
* No singing in the acid rain takes can quite.

You can find the notebook and data to create this project on my [Github](https://github.com/lukaszamora/Rush-Lyric-Generation/) page.

----