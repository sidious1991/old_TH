from __future__ import division
import got
import pickle
import tweepy
import time
import datetime

class TwittersRetweets:
    
    def __init__(self, since, until, query, twittapi):
        self.__since = since
        self.__until = until
        self.__query = query
        self.__twittapi = twittapi
        self.__twittersfilepath = '../outcomes/'+query+'#twitters.pickle' # default paths
        self.__tweetsfilepath = '../outcomes/'+query+'#tweets.pickle'
        self.__retweetsfilepath = '../outcomes/'+query+'#retweets.pickle'
    
    def setSince(self, since):
        self.__since = since
        return self
    
    def setUntil(self, until):
        self.__until = until
        return self
    
    def setQuery(self, query):
        self.__query = query
        return self
    
    def setTwittApi(self, twittapi):
        self.__twittapi = twittapi
        return self
    
    def setTwittersFilePath(self, path):
        self.__twittersfilepath = path
        return self
    
    def setTweetsFilePath(self, path):
        self.__tweetsfilepath = path
        return self
    
    def setRetweetsFilePath(self, path):
        self.__retweetsfilepath = path
        return self
    
    def getQuery(self):
        return self.__query
    
    '''
    @return: dictioTwitters, that is a dictionary like {username:{tweetcount:..},username:{tweetcount:..}...},
             of users who tweeted about the query and in the observation period specified.
    '''
    '''
    def computeTwitters(self):
        
        tweetCriteria = got.manager.TweetCriteria().setSince(self.__since).setUntil(self.__until).setQuerySearch(self.__query)
        twitters = got.manager.TweetManager.getTweets(tweetCriteria)
        
        dictioTwitters = {}
        dictioRetweets = {}
        tweetids = []
        
        for twitter in twitters:
            tweetids.append({twitter.id : twitter.username})
            
            if dictioTwitters.has_key(twitter.username):
                dictioTwitters[twitter.username]['tweetcount'] += 1   
            else:
                dictioTwitters.update({twitter.username : {'tweetcount':1}})
        
        #serialization
        with open(self.__twittersfilepath,'wb') as handle:
            pickle.dump(dictioTwitters, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(self.__tweetsfilepath,'wb') as handle:
            pickle.dump(tweetids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(self.__retweetsfilepath, 'wb') as handle:
            pickle.dump(dictioRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return dictioTwitters
    '''
    '''
    @return: dictioRetweets, that is a dictionary like {(retweetuser,tweetuser):{retweetprob:..},...},
             of retweets about the query.
             
             Probability bugged !!!
    '''
    '''
    def computeRetweets(self):
        
        with open(self.__twittersfilepath,'rb') as handle:
            dictioTwitters = pickle.load(handle)
        with open(self.__tweetsfilepath,'rb') as handle:
            tweets = pickle.load(handle)
        with open(self.__retweetsfilepath, 'rb') as handle:
            dictioRetweets = pickle.load(handle)  
        
        i = 0
        
        while i<len(tweets):
            
            tweetkey = (tweets[i].keys())[0] # the current dictionary of tweet id contains olny one key (tweet id)
            tweetuser = tweets[i][tweetkey] # user who tweetted
            tweetcount = dictioTwitters[tweetuser]['tweetcount'] # his tweetcount about this topic
        
            try:
                list_statuses = self.__twittapi.retweets(tweetkey) # list of status objects of retweets
                    
                for status in list_statuses:
                    retweetuser = (status._json['user']['screen_name']) # user who retweetted
                        
                    if not dictioTwitters.has_key(retweetuser): # insert retweet user in dictioTwitters though 
                        dictioTwitters.update({retweetuser: {'tweetcount':0}}) # he has not tweetted on the specific topic
                            
                    if dictioRetweets.has_key((retweetuser,tweetuser)):
                        num = dictioRetweets[(retweetuser,tweetuser)]['retweetprob']*tweetcount + 1
                        dictioRetweets[(retweetuser,tweetuser)]['retweetprob'] = num/tweetcount
                        
                    else:
                        p = 1/tweetcount
                        dictioRetweets.update({(retweetuser,tweetuser):{'retweetprob':p}})
                        
                i += 1
        
            except(tweepy.error.RateLimitError):   
                print '###twitter rate limit error### sleeping 15 minutes...'
                #sleep and calls itself
                print datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                time.sleep(900)
                print 'awake'
                continue # i unchanged
            
        with open(self.__twittersfilepath,'wb') as handle:
            pickle.dump(dictioTwitters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.__retweetsfilepath, 'wb') as handle:
            pickle.dump(dictioRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #tweets list is unchanged
     
        return dictioRetweets
    '''
if __name__ == '__main__':
    pass