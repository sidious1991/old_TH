from __future__ import division
import networkx as nx
import numpy as np
import math
from buildRetweetGraph.endorsementgraph import EndorsementGraph
#import itertools
import matplotlib.pyplot as plt
from scipy import linalg
from networkx.algorithms.community.centrality import girvan_newman

'''
    Source : 'Reducing Controversy by Connecting Opposing Views' - Garimella et alii
'''

'''
    @param path: is the path to diGraph (if not None)
    @param graph: is a diGraph (if not None)
    @param comms: are the communities found by asyn_fluidc in computeData
    @param partition: is the partition of the nodes found by 'Girvan Newman' in computeData
    @param type_sorting: if 0: nodes of each community ordered by in_degree,
                         elif 1: nodes of each community ordered by ratio (in_degree)/(out_degree + 1)
                         elif 2: nodes of each community ordered by betweenness centrality
                         else: nodes of each community ordered by AvgInDegree 
    @return sorted nodes of the two communities (sorted_x,sorted_y) by type_sorting
'''
def sortNodes(path, graph, comms, partition, type_sorting):
    
    if path is None and graph is None:
        return ()
    
    g = nx.read_gpickle(path) if path is not None else graph
    
    degrees_x = []
    degrees_y = []
    sorted_x = []
    sorted_y = []
    
    #in_degree     
    if type_sorting == 0:
        
        degrees_x = g.in_degree(comms[0]) #comm X -- to be ordered
        degrees_y = g.in_degree(comms[1]) #comm Y -- to be ordered
    
        sorted_x = sorted(degrees_x ,key=lambda tup: tup[1], reverse=True)
        sorted_y = sorted(degrees_y ,key=lambda tup: tup[1], reverse=True)
    
    #ratio
    elif type_sorting == 1:
        for i in comms[0]:
            degrees_x.append((i,g.in_degree(i)/(g.out_degree(i)+1)))
            
        for j in comms[1]:
            degrees_y.append((j,g.in_degree(j)/(g.out_degree(j)+1)))
            
        sorted_x = sorted(degrees_x ,key=lambda tup: tup[1], reverse=True)
        sorted_y = sorted(degrees_y ,key=lambda tup: tup[1], reverse=True)
        
    #betweenness centrality
    elif type_sorting == 2:
        centrality_x = nx.betweenness_centrality_subset(g, comms[0], comms[1], normalized=True)
        centrality_y = nx.betweenness_centrality_subset(g, comms[1], comms[0], normalized=True)
    
        sorted_x = sorted([i for i in centrality_x.iteritems() if partition[i[0]] == 0], key=lambda (k,v):(v,k), reverse=False)
        sorted_y = sorted([i for i in centrality_y.iteritems() if partition[i[0]] == 1], key=lambda (k,v):(v,k), reverse=False)

    else:
        for i in comms[0]:
            degrees_x.append((i,AvgInDegree(i, g)))
            
        for j in comms[1]:
            degrees_y.append((j,AvgInDegree(j, g)))

        sorted_x = sorted(degrees_x ,key=lambda tup: tup[1], reverse=True)
        sorted_y = sorted(degrees_y ,key=lambda tup: tup[1], reverse=True)

    return (sorted_x, sorted_y)

'''
    @param path: is the path to diGraph (if not None)
    @param graph: is a diGraph (if not None)
    @param type_sorting: if 0: nodes of each community ordered by in_degree,
                         elif 1: nodes of each community ordered by ratio (in_degree)/(out_degree + 1)
                         elif 2: nodes of each community ordered by betweenness centrality
                         else: nodes of each community ordered by AvgInDegree 
    @param percent_community: is the percentage of the high in_degree vertices to consider in each community
    @param a: is the probability to continue (1 - a is the restart probability)
    @return the communities of the graph, the personalization vectors for the communities,
            the c_x and c_y vectors, the partition and mats_x, mats_y tuple from M method,
            the sorted_x and sorted_y nodes of communities (by degree)
'''
def computeData(path, graph, a, type_sorting, percent_community = 0.25):
    
    if (path is None and graph is None):
        return ()
    
    g = nx.read_gpickle(path) if path is not None else graph
    
    comms = {}
    partition = {}
    i = 0
       
    comp = girvan_newman(nx.to_undirected(g))
    t = tuple(sorted(c) for c in next(comp))
    
    for c in t:
        comms.update({i:c})
        for node in c:
            partition.update({node:i})
        i+=1
        
    num_x = len(comms[0])
    num_y = len(comms[1])
    p_x = 1/num_x
    p_y = 1/num_y
    
    k1 = int(math.ceil(num_x*percent_community))
    k2 = int(math.ceil(num_y*percent_community))
    
    e_x = {}
    e_y = {}
    
    for key in comms.keys():
        for node in comms[key]:
            if key == 0:
                e_x.update({node: p_x})
                e_y.update({node: 0})             
            else:
                e_x.update({node: 0})
                e_y.update({node: p_y})
    
    c_x = []
    c_y = []
    
    degrees = g.in_degree(g.nodes())
        
    (sorted_x,sorted_y) = sortNodes(None, g, comms, partition, type_sorting)
    
    #inizialization
    for i in range(0,len(degrees)):
        c_x.append(0)
        c_y.append(0)
    
    minimum_k1 = min(k1,len(sorted_x))
    minimum_k2 = min(k2,len(sorted_y))
    
    for i in range(0,minimum_k1):
        c_x[sorted_x[i][0]] = 1
    for i in range(0,minimum_k2):
        c_y[sorted_y[i][0]] = 1
    
    mats_x = M(None, g, a, e_x)
    mats_y = M(None, g, a, e_y)
    
    return (e_x,e_y,c_x,c_y,mats_x,mats_y,comms,partition,sorted_x,sorted_y)
    

'''
    @param path: is the path to diGraph (if not None)
    @param l: is the sorted list which contains new edges to add with their delta_RWC*link_predictor
              i.e. l = [((node_from, node_to),link_predictor*delta_rwc), ((node_from, node_to),link_predictor*delta_rwc),..]
    @param dictio: dictio version of the list l with the information of link_predictor too
              i.e. dictio = {(edge):(link_predictor*delta_rwc,link_predictor), ..}
    @return new graph,total optimum delta RWC,ratio of accepted edges/proposed edges,maximum optimum delta RWC

'''
def addEdgeToGraph(path, l, dictio):
    
    if path is None:
        return ()
        
    g = nx.read_gpickle(path)
    
    delta=0
    max_delta = 0 # maximum expected delta
    count=0
    for i in range(0,len(l)):
        
        edge = l[i][0]
        
        delta_dot_predictor = l[i][1]
        
        pred = dictio[edge][1] #predictor for that edge
        #if pred == 0.0:
            #pred = 1.0
        
        delta += delta_dot_predictor/pred
        max_delta = min(max_delta,(delta_dot_predictor/pred))
        print delta
        #print edge
        
        g.add_edge(edge[0],edge[1])
        count +=1
        
    return (g,-delta,count/len(l),-max_delta)

    
'''
    @param path: is the path to diGraph (if not None)
    @param graph: is a diGraph (if not None) 
    @param a: is the probability to continue (1 - a is the restart probability)
    @param personal: is the restart vector
    @return the M_x or M_y matrix inverted (source Garimella et alii), depending on the restart vector (personal),
            and P_x or P_y matrix (transposed transition matrices)
'''
def M(path, graph, a, personal):
    
    if path is None and graph is None:
        return
    
    g = nx.read_gpickle(path) if path is not None else graph
    
    P = nx.google_matrix(g, alpha=1, dangling=personal) # Transition matrix (per row). Returns NumPy array that is different from
                                                        # ndarray!
    
    P_transp = np.transpose(P)
    
    I = np.identity(len(g.nodes()))

    m = np.subtract(I,np.dot(a,P_transp))

    m_inv = linalg.inv(m)
        
    p_array = np.array(P_transp)

    return (m_inv,p_array)
    
    
'''
    @param g: the graph to consider
    @param edge: the edge to predict
    @return the 'Adamic Adar' index for the predicted edge
'''    
def AdamicAdarIndex(g, edge):  
    
    if g is None or edge is None:
        return
  
    g_undirect = nx.to_undirected(g)
    
    source = edge[0]
    dest = edge[1]
    common = nx.common_neighbors(g_undirect, source, dest)
    
    index = 0.0
    
    for neigh in common:
        index += 1/math.log(g_undirect.degree(neigh), 10)
    
    index += 1 #To avoid zero indices, we scale them
    
    return index
   
'''
    @param g: the digraph
    @param node: the node to consider
    @return the average number of predecessors of node in the digraph g that retweet it 
''' 
def AvgInDegree(node, g):    
    
    preds = g.predecessors(node)
    avg_in_degree = 0.0
        
    for pred in preds:
        avg_in_degree += 1/g.out_degree(pred)                            
      
    return avg_in_degree
    
'''
    @param path: is the path to diGraph (if not None)
    @param graph: is a diGraph (if not None) 
    @param edge: directed edge to recommend (tuple: (to,from))
    @param data: tuple returned by computeData
    @return acceptance probability of the directed edge proposed

    Source (adapted version of):
    Balancing information exposure in social networks 
    Garimella, Parotsidis et alii
    
    Acceptance Probability, not based on polarization score
'''
'''
def acceptanceProbabilityGP(path, graph, edge, data):
    
    if path is None and graph is None:
        return
    
    g = nx.read_gpickle(path) if path is not None else graph
    
    (e_x,e_y,c_x,c_y,mats_x,mats_y,comms,part,sorted_x,sorted_y) = data

    alpha = 0.5 #look at the paper
    r_u = g.out_degree(edge[0]) # to adjust ... It is the total number of retweets of user u
    if r_u == 0:
        r_u += 1
    comm_v = part[edge[1]] # to which community the destination node v belongs
    
    count_comm_v = 0 # to adjust ... number of neighbors of u which belongs to comm_v 
    
    for node in g[edge[0]].keys():
        if part[node]==comm_v:
            count_comm_v += 1
    
    q_u = count_comm_v/r_u # a priori probability of a user u retweeting comm_v
    
    #r_u_v must be 0 in our case (we are proposing inexistent edge)...
    
    return alpha*q_u + (1-alpha)*(1/(r_u+2))
    
'''   
'''
    @param pscores: is a tuple identifying the polarity scores of a directed edge (from,to).
    @return the acceptance probability of the directed edge (from,to).
            The probability is based on the dipole and work concepts.
    future: total_acceptance_probability = alpha*P(work,dipole)+beta*P_other(other_factors)
            alpha + beta = 1
'''
'''
def acceptanceProbability(pscores):
    
    fromp = pscores[0]
    top = pscores[1]
    
    work = -np.exp(-(top))+np.exp(-(fromp)) if (fromp <= top) else np.exp(fromp)-np.exp(top)
    
    prob = np.exp(-work)
    
    return prob
''' 
'''
    @param path: is the path to diGraph (if not None)
    @param graph: is a diGraph (if not None) 
    @param data: tuple returned by computeData
    @return polarization score for each node in diGraph.
            Algorithm used (adapted) from source:
            
            Measuring Political Polarization: Twitter shows the two sides of Venezuela
            A. J. Morales, J. Borondo, J. C. Losada and R. M. Benito
            Grupo de Sistemas Complejos. Universidad Politecnica de Madrid.
            ETSI Agronomos, 28040, Madrid, Spain
            
            This algorithm is based on LABEL-PROPAGATION
'''
'''
def polarizationScore(path, graph, data):
    
    if path is None and graph is None:
        return
    
    g = nx.read_gpickle(path) if path is not None else graph
    
    (e_x,e_y,c_x,c_y,mats_x,mats_y,comms,part,sorted_x,sorted_y) = data
    
    alpha = 1e-3 #if the error is sufficiently small stop iterations
    tocontinue = True #true if necessary other iterations (error > alpha for some node)
    
    #we take the top 5% of nodes of X community (ordered in decreasing order of %in_degree) as "Elite Nodes"
    num_top_x = int(math.ceil(0.05*len(comms[0])))
    #we take the top 5% of nodes of Y community (ordered in decreasing order of %in_degree) as "Elite Nodes"
    num_top_y = int(math.ceil(0.05*len(comms[1])))
    dictio_polarization = {0:{}}
    ratio_x = {}
    ratio_y = {}
    
    for i in comms[0]:
        print i
        ratio_x.update({i:g.in_degree(i)/g.degree(i)})
    
    for i in comms[1]:
        print i
        ratio_y.update({i:g.in_degree(i)/g.degree(i)})
        
    ratio_x_ordered = sorted(ratio_x.iteritems(), key=lambda (k,v):(v,k), reverse=True)
    ratio_y_ordered = sorted(ratio_y.iteritems(), key=lambda (k,v):(v,k), reverse=True)
    
    #initialization
    for i in range(0,len(ratio_x_ordered)):
        if i < num_top_x:
            dictio_polarization[0].update({ratio_x_ordered[i][0]:ratio_x_ordered[i][1]})
        else:
            dictio_polarization[0].update({ratio_x_ordered[i][0]:0})
            
    for i in range(0,len(ratio_y_ordered)):
        if i < num_top_y:
            dictio_polarization[0].update({ratio_y_ordered[i][0]:-ratio_y_ordered[i][1]})
        else:
            dictio_polarization[0].update({ratio_y_ordered[i][0]:0})
    
    row_order = g.nodes()
    adj = nx.attr_matrix(g,rc_order=row_order)
    adj_array = np.array(adj)#numpy
    
    iter = 1
    #iterations
    while tocontinue:
        
        tocontinue = False
        dictio_polarization.update({iter:{}})
        
        for j in range(0,len(ratio_x_ordered)):
            
            if j >= num_top_x:
                row = adj_array[ratio_x_ordered[j][0]]
                sum_pol = 0
                for k in row_order:
                    sum_pol += row[k]*dictio_polarization[iter-1][k]
                    
                newval = sum_pol/g.out_degree(ratio_x_ordered[j][0])
                oldval = dictio_polarization[iter-1][ratio_x_ordered[j][0]]
                dictio_polarization[iter].update({ratio_x_ordered[j][0]:(newval)})   
                
                tocontinue = math.fabs((newval-oldval)/oldval) > alpha if oldval != 0 else (math.fabs(newval-oldval)) > alpha
                
            else:
                dictio_polarization[iter].update({ratio_x_ordered[j][0]:ratio_x_ordered[j][1]})
        
        for j in range(0,len(ratio_y_ordered)):
            
            if j >= num_top_y:
                row = adj_array[ratio_y_ordered[j][0]]
                sum_pol = 0
                for k in row_order:
                    sum_pol += row[k]*dictio_polarization[iter-1][k]
                    
                newval = sum_pol/g.out_degree(ratio_y_ordered[j][0])
                oldval = dictio_polarization[iter-1][ratio_y_ordered[j][0]]
                dictio_polarization[iter].update({ratio_y_ordered[j][0]:(newval)})   
                
                tocontinue = math.fabs((newval-oldval)/oldval) > alpha if oldval != 0 else (math.fabs(newval-oldval)) > alpha
 
            else:
                dictio_polarization[iter].update({ratio_y_ordered[j][0]:-ratio_y_ordered[j][1]})     
 
        iter += 1
    
    return dictio_polarization[iter-1]
'''
if __name__ == '__main__':
    '''
    graphData = computeData('../../outcomes/parted_graph.pickle', None, 40, 0.85)
    
    pol = polarizationScore('../../outcomes/parted_graph.pickle', None, graphData)
    print pol
    
    tuple = sortNodes('../../outcomes/parted_graph.pickle', None, 4)
    print tuple[0]
    print tuple[1]
    
    p = acceptanceProbabilityGP('../../outcomes/parted_graph.pickle', None, (50,109), graphData)
    print p
    '''    
    eg = EndorsementGraph("retweet_graph_beefban")
    g = eg.buildEGraph()
    print g.edges(data = False)
    print len(g.edges())
    print g.nodes(data = False)
    print len(g.nodes)
    nx.draw_networkx(g, labels = {21:21, 418:418, 439:439})
    plt.show()
    