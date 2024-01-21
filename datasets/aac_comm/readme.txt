A Crowdsourced Corpus of AAC-like Communications
==================================================
http://www.aactext.org/imagine/

We used Amazon Mechanical Turk to create a large set of fictional AAC-like 
communications.  Workers were asked to invent communications as if they were 
using a scanning-style interface to communicate.  Our corpus contains 
approximately six thousand communications.  We found our crowdsourced 
collection modeled conversational AAC better than datasets based on telephone 
conversations or newswire text.  We leveraged our crowdsourced messages to 
intelligently select sentences from much larger sets of Twitter, blog and 
Usenet data. 

For details, see the paper "The Imagination of Crowds: Conversational AAC 
Language Modeling using Crowdsourcing and Large Data Sources", in EMNLP 2011:
http://www.keithv.com/pub/imagine/imagine_aac_lm.pdf

We have included the cleaned crowdsourced communications. These files include 
the original case and punctuation used by the workers.  We have also included 
some of the word lists we used to build our language models.

sent_train_aac.txt     TurkTrain training set, from 80% of the workers
sent_dev_aac.txt       TurkDev test set, from 10% of the workers
sent_test_aac.txt      TurkTest test set, from 10% of the workers 

vocab_aac_twitter.txt  63K word vocabulary used by our language models
american_words.txt     330K American English words, used to limit our language 
                       model vocabulary to avoid common misspellings, etc.

lm_test_switch.txt     SwitchTest test set, 59 Switchboard sentences
lm_test_comm.txt       Comm test set, 251 sentences written in response to 
                       hypothetical communication situations. Originally 
                       collected by H. Venkatagiri in the paper "Efficient 
                       Keyboard Layouts for Sequential Access in Augmentative 
                       and Alternative Communication".                    

A selection of the trained AAC language models can be found at:
http://www.aactext.org/imagine/

Have fun!
Keith Vertanen
Per Ola Kristensson

Revision history:
==================================================
7/2011 - Initial release of the crowdsourced AAC communication corpus.