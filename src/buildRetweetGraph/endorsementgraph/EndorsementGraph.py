from __future__ import division
import networkx as nx
import nxmetis
import matplotlib.pyplot as plt
import pickle

class EndorsementGraph:
    
    def __init__(self, inputname):
        self.__inputname = inputname

    def setInputName(self, inputname):
        self.__inputname = inputname
        
    def getInputName(self):
        return self.__inputname
    
    def buildEGraph(self):
        inputname = self.getInputName()
        digraph = nx.DiGraph(inputname = inputname)
        
        current_val = 0
        dictio_nodes_convert = {} # node : node_index_in_graph (key : value) -- convert each node string in integer id
        dictio_nodes = {} # node : totalretweetcount (key : value)
        dictio_edges = {} # (source, dest) : retweetcount (key : value)
        
        #Build digraph read from file and write back pickled
        with open ('../../inputs/'+inputname+'.txt') as f:
            for line in f:
                l = line.split(',')
                source = l[0]
                dest = l[1]
                rtwcnt = float(l[2].rstrip('\n'))
                                
                if dictio_nodes.has_key(source):
                    dictio_nodes[source] += rtwcnt
                
                else:
                    dictio_nodes.update({source : rtwcnt})
                    dictio_nodes_convert.update({source : current_val})
                    current_val += 1
                
                if not dictio_nodes.has_key(dest):
                    dictio_nodes.update({dest : 0})
                    dictio_nodes_convert.update({dest : current_val})
                    current_val += 1
                
                dictio_edges.update({(source, dest) : rtwcnt})
    
            #Set nodes of the endorsement graph
            for key in dictio_nodes.keys():
                digraph.add_node(dictio_nodes_convert[key], totalretweetcount = dictio_nodes[key])
                
            #Set edges of the endorsement graph
            for key in dictio_edges.keys():
                digraph.add_edge(dictio_nodes_convert[key[0]], dictio_nodes_convert[key[1]], prob = dictio_edges[key]/dictio_nodes[key[0]])
                
        #serialization
        nx.write_gpickle(digraph, '../../outcomes/'+inputname+'.pickle', protocol=pickle.HIGHEST_PROTOCOL)
        
        return digraph
    
    '''
    def __init__(self, twrtw):
        self.__twrtw = twrtw
        self.__graphfilepath = '../outcomes/'+twrtw.getQuery()+'#digraph.pickle' #default path
        
    def setEGraphFilePath(self, path):
        self.__graphfilepath = path
        return self
    
    def setTwrRtw(self, twrtw):
        self.__twrtw = twrtw
        return self
    '''
    '''
    @return: a digraph representing the endorsement graph about the query in the observation period
             specified in self.__twrtw. The digraph is serialized by pickle.
             
             Based on bugged code!!! (computeRetweets)
    '''
    '''
    def buildEGraph(self):
        
        digraph = nx.DiGraph(topic = self.__twrtw.getQuery());
        
        dictioTwitters = self.__twrtw.computeTwitters()
        dictioRetwitters = self.__twrtw.computeRetweets()
        
        for key in dictioTwitters.keys():
            digraph.add_node(key, tweetcount = dictioTwitters[key]['tweetcount'])# a node of graph has attribute tweetcount
            
        for key in dictioRetwitters.keys():
            #Garimella et al. consideration (minimum number of retweets by key[0] for key[1] content)
            if digraph.node[key[1]]['tweetcount']*dictioRetwitters[key]['retweetprob'] >= 2:
                digraph.add_edge(key[0], key[1], retweetprob=dictioRetwitters[key]['retweetprob'])    
         
        #Delete all the nodes with degree (outdegree+indegree) == 0
        for node in digraph.nodes().keys():
            if digraph.degree(node) == 0:
                digraph.remove_node(node)
                 
        #serialization
        nx.write_gpickle(digraph, self.__graphfilepath, protocol=pickle.HIGHEST_PROTOCOL)
        
        return digraph
    '''
          
if __name__ == "__main__":
    pass
    
    
    