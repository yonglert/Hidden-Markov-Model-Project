####
# Team SPGMA
# Members and Duties
# Lee Ying Yang A0170208N - Naive 1, Viterbi 1, Forwards Backwards, Cat Predict
# Tang Yong Ler A0199746E - Naive 2, Forwards Backwards, Cat Predict
# Alvin Tan Jia Liang A0203011L - Viterbi 2, Forwards Backwards, Cat Predict
# Lin Da A0201588A - Viterbi 2, Forwards Backwards, Cat Predict
####

# import library
import numpy as np
from collections import Counter
import re
from itertools import permutations
# import os
# os.chdir('../Users/yonglertang/Documents/GitHub/SPGMA')


# helper read text functions
def read_training_text(text_file_name):
    '''
    return list of tuples (token, tag)
    '''
    
    with open(text_file_name) as f:
        data = f.read()
     
    #split by double \n\n which indicates blank line    
    data = data.split("\n\n")
    if data[-1]=='':
        del data[-1]

    data = [[tuple(row.split("\t")) for row in chunk.split("\n")] 
            for chunk in data]
    return data

def read_test_text(text_file_name):
    '''
    return list of tokens
    '''
    
    with open(text_file_name) as f:
        data = f.read()
    data = data.split("\n\n")
    if data[-1]=='':
        del data[-1]
    data = [chunk.split("\n") for chunk in data]
    return data

def read_states_text(state_textfile):
    '''
    read twitter_tags, return list of tags
    '''
    with open(state_textfile) as f:
        data = f.read()
    data = data.split("\n")
    if data[-1]=='':
        del data[-1]
    return data


def read_prob_file(prob_file):
    '''
    return dictionary of { (token, state) : p(token | state) } [emission prob]
    read in format of x\ty\tp each row, 
    \t -> tab seperate
    x -> token
    y -> state
    p -> prob
    such that p = p(x|y) 
    returns dict with (x,y) as key, p as value. 
    {(x,y):p}
    '''
    with open(prob_file) as f:
        data = f.read()
    data = data.split("\n")
    if data[-1]=='':
        del data[-1]
    data = (i.split("\t") for i in data)
    data = dict([*map(lambda x: ((x[0],x[1]),float(x[2])),data)])
    return data


# prob generation function 

smooth_value = [0.01, 0.1, 1, 10]
def generate_output_probs_1(training_file, states_file, 
                           output_prob_file, smooth_value = smooth_value[1]):
    '''
    Naive prediction 1 approach, generate probs naively by counting y = j -> x = w / count (y = j)
    Then use smoothing value of 0.1 to smooth and ensure no zero probability.
    i.e prob(token | state) = count(y=j -> x=w)
    '''

    data = read_training_text(training_file)
    states= read_states_text(states_file)
    token_state_count = Counter()
    state_count = Counter()
    token_set = set()
    token_set.add("-UNK-TOKEN-") #unknown
    #get counts
    for chunk in data:
        chunk_states = (i[1] for i in chunk)
        chunk_words = (i[0] for i in chunk)

        token_state_count.update(chunk)
        token_set.update(chunk_words)
        state_count.update(chunk_states)
    n_words = len(token_set)
    smooth_value_denom = (n_words+1)*smooth_value
    with open(output_prob_file,"w") as f:
        for token in token_set:
            for state in states:
                n_token_state = token_state_count[(token,state)]
                prob_token_state = (n_token_state+smooth_value)/ \
                    (state_count[state]+smooth_value_denom)
                f.writelines(f"{token}\t{state}\t{prob_token_state}\n")
#generate_output_probs_1("twitter_train.txt","twitter_tags.txt","naive_output_probs.txt")


def auto_classify(word):
    if re.search('^@',word):
        return "-@mention-"
    elif re.search('^#',word):
        return "-#hashtag-"
    elif re.search(r'((http|ftp|https)(:\/\/))?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',word):
        return "-httplink-"
    # elif re.search(r'\$?[0-9]+',word):
    #     return word
    else:
        return word.lower()



def generate_output_probs_2(training_file, states_file, 
                           output_prob_file, smooth_value = smooth_value[1]):
    '''
    generate naively p(x=w|y=j) as
    count(y = j -> x = w)/count(y = j)
    outputs to format of x\ty\tp(x = w|y = j) for each row
    '''
    data = read_training_text(training_file)
    states= read_states_text(states_file)
    token_state_count = Counter()
    state_count = Counter()
    token_set = set()
    token_set.add("-UNK-TOKEN-")
    #get counts
    for chunk in data:
        chunk_states = (i[1] for i in chunk)
        chunk_words = (auto_classify(i[0]) for i in chunk)
        chunk = ((auto_classify(i),j) for i,j in chunk)
        token_state_count.update(chunk)
        token_set.update(chunk_words)
        state_count.update(chunk_states)
    n_words = len(token_set)
    smooth_value_denom = (n_words+1)*smooth_value
    with open(output_prob_file,"w") as f:
        for token in token_set:
            for state in states:
                n_token_state = token_state_count[(token,state)]
                prob_token_state = (n_token_state+smooth_value)/ \
                    (state_count[state]+smooth_value_denom)

                f.writelines(f"{token}\t{state}\t{prob_token_state}\n")


#generate_output_probs_2("twitter_train.txt","twitter_tags.txt","naive_output_probs.txt")

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    '''
    Using naively generated bj(w) which is P(x = w | y = j)
    we find best tag j* which is argmax bj(w) = argmax P(x = w |y =j)
    Accuracy -> 64.8%
    '''
    p_x_given_y = read_prob_file(in_output_probs_filename)
    states = set((key[1] for key in p_x_given_y.keys()))
    known_tokens = set((key[0] for key in p_x_given_y.keys()))
    states = list(states)
    test_data = read_test_text(in_test_filename)
    with open(out_prediction_filename,"w") as f:
        for chunk in test_data:
            for token in chunk:
                token = token if token in known_tokens else "-UNK-TOKEN-"
                token_probs = sorted([[p_x_given_y[(token,state)],state] for state in states],key=lambda x:x[0])

                max_prob_state = token_probs[-1][1]
                f.writelines(f"{max_prob_state}\n")
            f.writelines(f"\n") #extra newline per tweet

#naive_predict("naive_output_probs.txt","twitter_dev_no_tag.txt","naive_prediction.txt")


def naive_predict2(in_output_probs_filename,in_train_file_name, in_test_filename, out_prediction_filename):
    '''
    Naive prediction 2 approach: argmax P(y = j | x = w) to find j*
    i.e given the word, what is the most suitable tag for the word?
    to get P(y = j | x = w), we find using P(x = w | y = j) * p(y = j) / p(x = w) aka Bayes Rule
    p(y = j) -> count(y = j) / total count of Y
    p(x = w) -> count(x = w) / total count of X
    then find argmax...
    Accuracy -> 69.4% 
    '''
    p_x_given_y = read_prob_file(in_output_probs_filename)
    states = set((key[1] for key in p_x_given_y.keys()))
    known_tokens = set((key[0] for key in p_x_given_y.keys()))
    states = list(states)
    test_data = read_test_text(in_test_filename)

    ####################################
    data = read_training_text(in_train_file_name)
    word_count = Counter()
    state_count = Counter()
    for chunk in data:
        chunk_states = (i[1] for i in chunk)
        chunk_words = (i[0] for i in chunk)
        word_count.update(chunk_words)
        state_count.update(chunk_states)
    n_words = sum(word_count.values())
    n_states = sum(state_count.values())
    SMOOTH_VALUE = 0.1
    smooth_value_denom_x = (n_words + 1) * SMOOTH_VALUE
    smooth_value_denom_y = (n_states + 1) * SMOOTH_VALUE
    ####################################

    with open(out_prediction_filename,"w") as f:
        for chunk in test_data:
            for token in chunk:
                token = token if token in known_tokens else "-UNK-TOKEN-"
                num_x = word_count[token] + 1
                p_x = (num_x * SMOOTH_VALUE) / smooth_value_denom_x
                token_probs = sorted([[(p_x_given_y[(token,state)] * \
                    (((state_count[state]+1)*SMOOTH_VALUE)/smooth_value_denom_y)) / p_x ,state] for state in states],key=lambda x:x[0])
                max_prob_state = token_probs[-1][1]
                f.writelines(f"{max_prob_state}\n")
            f.writelines(f"\n") #extra newline per tweet

#naive_predict2("naive_output_probs.txt", 'twitter_train.txt','twitter_dev_no_tag.txt', 'naive_predictions2.txt')


def generate_trans_prob(training_file, states_file, output_prob_file, smooth_value=0.1):
    '''
    outputs trans prob in the form of yt yt-1 p seperated by \t
    end state will be marked by -END-STATE-
    state state will be marked by -START-STATE-
    '''
    data = read_training_text(training_file)
    states = read_states_text(states_file)
    chunk_states = [["-START-STATE-",*(j for _,j in chunk),"-END-STATE-"] for chunk in data]
    possible_permutations = [*permutations(states,2),
                             *(("-START-STATE-",i) for i in states),
                             *((i,"-END-STATE-") for i in states),
                             *((i,i) for i in states)
                             ]
    possible_permutations.sort()
    transition_counter = Counter()
    state_counter = Counter()
    for chunk in chunk_states:
        chunk_len = len(chunk)
        chunk_transitions = [(chunk[i],chunk[i+1]) for i in range(chunk_len-1)]
        state_counter.update(chunk)
        transition_counter.update(chunk_transitions)
    smooth_value_denom = (len(states)+2+1)*smooth_value
    with open(output_prob_file,"w") as f:
        for transition in possible_permutations:
            cur_state,next_state = transition
            trans_count = transition_counter[transition]
            prob = (trans_count+smooth_value)/(state_counter[cur_state]+smooth_value_denom)
            f.writelines(f"{next_state}\t{cur_state}\t{prob}\n")

#generate_output_trans_prob("twitter_train.txt","twitter_tags.txt", "trans_probs.txt")


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):

    states = read_states_text(in_tags_filename)
    trans_probs = read_prob_file(in_trans_probs_filename)
    emission_probs = read_prob_file(in_output_probs_filename)
    test_data = read_test_text(in_test_filename)

    known_tokens = set((key[0] for key in emission_probs.keys()))
    with open(out_predictions_filename,"w") as f:
        for chunk in test_data:

            #first for loop for 1 step from start to state x, x = 1,..,N
            Π=[]
            backpointers =[]
            token = chunk[0]
            token = token if token in known_tokens else "-UNK-TOKEN-"
            pi_0 = [trans_probs[(state,"-START-STATE-")]*emission_probs[(token,state)] for state in states]
            Π.append(pi_0)

            #second for loop for steps from state 2 to n

            for i in range(1,len(chunk)): # for each time step
                token = chunk[i]
                token = token if token in known_tokens else "-UNK-TOKEN-"
                pi_i = []
                cur_pointers = []
                for cur_state in states: #for each state in that time step
                    # all Π(i,j)a(states,j)bj(x), sorted
        
                    # finding maximum path
                    prob = sorted([(Π[i-1][j]*trans_probs[(cur_state,states[j])]*
                            emission_probs[(token,cur_state)],j) for j in range(len(states))],
                            key=lambda x:x[0])
                    max_prob = prob[-1][0]              # append prob to table
                    max_prob_pointer = prob[-1][1]      # prev state that gives best prob for cur state
                    pi_i.append(max_prob)                       # add prob
                    cur_pointers.append(max_prob_pointer)       # add backpointer

                Π.append(pi_i)
                backpointers.append(cur_pointers)
            
            pi_last = [(trans_probs[("-END-STATE-",state)]*Π[-1][i],i) for i,state in enumerate(states)]
            last_tag = sorted(pi_last)[-1][1]
            #backwards traverse
            pred_tags = [last_tag]
            for i in range(len(backpointers)-1,-1,-1):
                pred_pointer = backpointers[i][pred_tags[-1]]
                pred_tags.append(pred_pointer)
            pred_tags = pred_tags[::-1]
            pred_tags = [states[i] for i in pred_tags]
            pred_tags = "\n".join(pred_tags)
            f.writelines(pred_tags)
            f.writelines("\n\n")


# viterbi_predict("twitter_tags.txt", "trans_probs.txt","naive_output_probs.txt",
#                 "twitter_dev_no_tag.txt","viterbi_predictions.txt")
                
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    states = read_states_text(in_tags_filename)
    trans_probs = read_prob_file(in_trans_probs_filename)
    emission_probs = read_prob_file(in_output_probs_filename)
    test_data = read_test_text(in_test_filename)
    known_tokens = set((key[0] for key in emission_probs.keys()))

    with open(out_predictions_filename,"w") as f:
        for chunk in test_data:

            #first for loop for 1 step from start to state x, x = 1,..,N
            Π=[]
            backpointers =[]
            token = auto_classify(chunk[0])
            token = token if token in known_tokens else "-UNK-TOKEN-"

            pi_0 = [trans_probs[(state,"-START-STATE-")]*emission_probs[(token,state)] for state in states]
            Π.append(pi_0)

            #second for loop for steps from state 2 to n

            for i in range(1,len(chunk)): # for each time step
                token = auto_classify(chunk[i])

                token = token if token in known_tokens else "-UNK-TOKEN-"
                pi_i = []
                cur_pointers = []
                for cur_state in states: #for each state in that time step
                    # all Π(i,j)a(states,j)bj(x), sorted
        
                    # finding maximum path
                    prob = sorted([(Π[i-1][j]*trans_probs[(cur_state,states[j])]*
                            emission_probs[(token,cur_state)],j) for j in range(len(states))],
                            key=lambda x:x[0])
                    max_prob = prob[-1][0]              # append prob to table
                    max_prob_pointer = prob[-1][1]      # prev state that gives best prob for cur state
                    pi_i.append(max_prob)                       # add prob
                    cur_pointers.append(max_prob_pointer)       # add backpointer

                Π.append(pi_i)
                backpointers.append(cur_pointers)
            
            pi_last = [(trans_probs[("-END-STATE-",state)]*Π[-1][i],i) for i,state in enumerate(states)]
            last_tag = sorted(pi_last)[-1][1]
            #backwards traverse
            pred_tags = [last_tag]
            for i in range(len(backpointers)-1,-1,-1):
                pred_pointer = backpointers[i][pred_tags[-1]]
                pred_tags.append(pred_pointer)
            pred_tags = pred_tags[::-1]
            pred_tags = [states[i] for i in pred_tags]
            pred_tags = "\n".join(pred_tags)
            f.writelines(pred_tags)
            f.writelines("\n\n")


# instead of value passing.
def get_alpha(tweet,trans_probs,output_probs,states):
    # calculate alpha_start, list of prob from -START-STATE- to each tag and outputting the word given the tag
    # i.e [P(-START-STATE- to "N") * P(1st_word | "N"), P(-START-STATE- to "V") * P(1st_word | "V"),...]
    alpha_1 = [trans_probs[("-START-STATE-",to_state)] * output_probs[(tweet[0],to_state)] \
        for to_state in states]
    alphas = [alpha_1]

    # calculate alpha_2 to alpha_n
    # alpha_t(current_tag) = sum(alpha_{t-1}(all_possible_tags) * trans_prob(all_possible_tags to current tag) * output_prob(word | current tag)
    for k in range(1,len(tweet)):
        cur_token = tweet[k] # current word in the tweet

        # TO GET alpha_t:
        # for each tag to all tags, get the trans_prob(prev_tag -> cur_tag) and output_prob(word | new_tag)
        # get alpha_{t-1} by indexing -1 from alphas (list of alphas)
        # sum them up
        cur_alpha = [sum([alphas[-1][i] * trans_probs[(from_state,to_state)] * output_probs[(cur_token,to_state)] for i,from_state in enumerate(states)])\
            for to_state in states]
        alphas.append(cur_alpha)

    # alpha_end
    alpha_end =  [alphas[-1][i] * trans_probs[(from_state,"-END-STATE-")] for i,from_state in enumerate(states)] 
    return np.array(alphas),np.array(alpha_end)

def get_beta(tweet,trans_probs,output_probs,states):
    # beta of 1 to n
    # beta 0 doesnt really seem to be needed

    beta_n = [trans_probs[(from_state,"-END-STATE-")] for from_state in states]
    betas = [beta_n]
    #fill in betas from back to front

    # iterate in descending order
    for k in range(len(tweet)-1, 0, -1):
        # beta_j(t+1) * bj(x_{t+1}) * ai,j
        # i is from state, j is next state
        succesive_token = tweet[k] # x(t+1)
        cur_beta = [sum([betas[-1][j] * trans_probs[(from_state, to_state)] * output_probs[(succesive_token, to_state)]\
            for j,to_state in enumerate(states)]) \
            for from_state in states]
        betas.append(cur_beta)

    beta_0 = [trans_probs[("-START-STATE-",to_state)] * output_probs[(tweet[0],to_state)] * betas[-1][j]\
                for j,to_state in enumerate(states)]

    #reverse betas to get correct front to back ordering
    betas = betas[ : :-1]
    return np.array(betas), np.array(beta_0)


def get_gamma(alphas,betas,alpha_end_sum):
    # returns (t,j) array
    return (alphas * betas) / alpha_end_sum
    

def get_xi(tweet,alphas,betas,alpha_end_sum,trans_probs,output_probs,states):
    #return (t-1,i,j) np array
    xi = []
    for t in range(len(tweet)-1):
        next_token = tweet[t+1]
        xi_t = []    
        for i,from_state in enumerate(states):
            xi_t_i = [alphas[t][i] * trans_probs[(from_state,to_state)] * output_probs[(next_token,to_state)] * betas[t+1][j]\
                        for j,to_state in enumerate(states)]
            xi_t.append(xi_t_i)
        xi.append(xi_t)
    xi = np.array(xi)
    xi /= alpha_end_sum 
    return xi

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    
    # Step 1, get all possible parameters to estimate, initialize it as random
    # Step 2a, alpha beta every tweet, every step
    # step 2b, gamma and zai per tweet, every step
    # step 3, m step. Update estimates for parameters  repeat till convergence

    # set seed for random generator
    np.random.seed(seed)

    # List of all possible tags (states)
    states = read_states_text(in_tag_filename)

    # Untagged tweets -> Generate a list of list of tweets.
    # i.e [["hello", "world", "!"], ["Happy", "Halloween", "everybody", "!", "@USER_0db49cdf"]]
    training_data = read_test_text(in_train_filename) 

    # trans_prob keys (from,to), value a(i,j)
    trans_probs = {}

    # output_prob keys (word,from), value bj(word)
    output_probs = {}

    # Create a list of possible tags we can ARRIVE FROM (possible_from_tags -> current_tag): possible_from
    # Create a list of possible tags we can GO TO (current_tag -> possible_to_tags): possible_to
    possible_from = states+["-START-STATE-"]  # possible to start from -START-STATE-
    possible_to = states+["-END-STATE-"] # possible to go to -END-STATE-

    # Initialize transition probability for each tags to every other tags/state (trans_probs)
    # -> for each tag, initialize a random probability value to every tag it can transit to
    # key : (curr_tag, next_tag), value : transition probability
    for from_state in possible_from:
        # start directly to end will imply empty sequence, thus omited
        if from_state == "-START-STATE-": 
            possible_to = states

        # initialization -> if there are 5 possible tags, generate np.array of 5 random values then normalize
        n_to_tags = len(possible_to)
        transition_prob = np.random.random(n_to_tags)
        transition_prob = transition_prob / transition_prob.sum() # normalize to 1 to obey axioms of probability

        # index: integer, to_state: tag. i.e there is 24 tags in total without -START-STATE-, -END-STATE-
        for index, to_state in enumerate(possible_to):
            # set the randomly initialised transition_prob of (from_tag -> to_tag) into trans_probs
            # trans_probs will have: 
            #     i.e @ -> @ : value, @ -> , : value, @ -> L : value...
            trans_probs[(from_state,to_state)] = transition_prob[index]

    # all possible token (i.e each word in a tweet is a token)
    # also preprocess training data
    token_set = set() # token_set stores all "seen words"
    for index, tweet in enumerate(training_data): 
        tweet = [auto_classify(token) for token in tweet] # auto classify each word in the tweet
        training_data[index] = tweet # update the training data with the auto_classified tweet
        token_set.update(tweet) # update the token set with words seen in each tweet

    token_set.add("-UNK-TOKEN-") # add an unknown token
    n_unique_tokens = len(token_set) # number of unique words from training data

    # Initialize output probability for each word to every tags/state
    # -> for each word, initialize a random probability value that the word is emitted given the state
    # key : (word, tag), value : output probability
    for from_state in possible_from:
        state_output = np.random.random(n_unique_tokens) 
        state_output = state_output / state_output.sum() # normalize all output prob given a state to be 1
        for i,word in enumerate(token_set): 
            output_probs[(word,from_state)] = state_output[i]
    
    def get_current_loglikelihood(training_data,trans_probs,output_probs,states):
        log_likelihood = 0
        for tweet in training_data:
            _,cur_alpha_end = get_alpha(tweet,trans_probs,output_probs,states)
            log_likelihood += np.log(cur_alpha_end.sum())
        return log_likelihood

    # calculate smoothing values for transitional prob
    # and for emission probability
    SMOOTH_VALUE = 0.0000001

    #number of states including start and end state and +1
    smooth_states_denom = (len(states) + 3) * SMOOTH_VALUE
    smooth_token_denom = (n_unique_tokens + 1) * SMOOTH_VALUE


    prev_log_likelihood = 0
    cur_log_likelihood = 0
    for step in range(1, max_iter + 1):
        print(f"iteration {step} started")
        prev_log_likelihood = cur_log_likelihood
        a_i_j = np.zeros((len(states),len(states)))
        start_prob = np.zeros(len(states))
        end_prob = np.zeros(len(states))
        b_numerator = {token:np.zeros(len(states)) for token in token_set}
        b_denom =  np.zeros(len(states))

        for tweet in training_data:
            cur_beta , cur_beta_0 = get_beta(tweet,trans_probs,output_probs,states)
            cur_alpha,cur_alpha_end = get_alpha(tweet,trans_probs,output_probs,states)
            alpha_end_sum = cur_alpha_end.sum()
            cur_gamma = get_gamma(cur_alpha, cur_beta, alpha_end_sum)
            cur_xi = get_xi(tweet,cur_alpha, cur_beta, alpha_end_sum,trans_probs,output_probs,states)
            

            # alpha_end_sum = p(x1,x2,x3,...xn)
            # aka, it's the likelihood of data, under current parameters
            # log and add to loglikelihood for 
            # total loglihood of data

            a_i_j += cur_xi.sum(axis = 0) # sum across time
            b_denom += cur_gamma.sum(axis = 0) # sum across time
            for t,token in enumerate(tweet):
                b_numerator[token] += cur_gamma[t,:]
            
            # use summation terms of beta 0 to estimate start probability
            # as they are equivalent to numerator of xi 0
#            start_prob+= cur_beta[0,:]*np.array([output_probs[(chunk[0],from_state)] for from_state in state])/alpha_end_sum
            start_prob += np.array([trans_probs[("-START-STATE-",to_state)] * output_probs[(tweet[0],to_state)] for to_state in states]) \
                         * cur_beta[0] / alpha_end_sum
            
            #use summation terms of alpha n+1 to estimate end probability
            end_prob += cur_alpha_end / alpha_end_sum


        a_i_j = np.concatenate([a_i_j, end_prob.reshape((-1,1))], axis=1) #newly added
        #normalize transitional prob and add smoothing
        a_i_j = (a_i_j + SMOOTH_VALUE) / (a_i_j.sum(axis=1).reshape((-1,1)) + smooth_states_denom)
        start_prob = (start_prob + SMOOTH_VALUE) / (start_prob.sum() + smooth_states_denom)
        end_prob = a_i_j[:,-1].reshape(-1) 

        for i,from_state in enumerate(states):
            for j,to_state in enumerate(states):
                trans_probs[(from_state,to_state)] = a_i_j[i,j]
            trans_probs[(from_state,"-END-STATE-")] = end_prob[i]
            trans_probs[("-START-STATE-",from_state)] = start_prob[i]
            for token in token_set:
                output_probs[(token,from_state)] = (b_numerator[token][i] + SMOOTH_VALUE) / (b_denom[i] + smooth_token_denom)
        
        cur_log_likelihood = get_current_loglikelihood(training_data,trans_probs,output_probs,states)
        print(f"Loglikelihood after iteration: {cur_log_likelihood}")
        fractional_improvement = abs((cur_log_likelihood - prev_log_likelihood) / prev_log_likelihood)
        if(fractional_improvement <= thresh and prev_log_likelihood != 0):
            print(f"Fractional improvement of loglikelihood {fractional_improvement:0.6f} <= thresh of {thresh}")
            print(f"Terminating iterations")
            break
        

    print(f"Final loglikelihood: {get_current_loglikelihood(training_data,trans_probs,output_probs,states)}")
    with open(out_trans_filename,"w") as f:
        for transition in trans_probs.keys():
            cur_state,next_state = transition
            prob = trans_probs[transition]
            f.writelines(f"{next_state}\t{cur_state}\t{prob}\n")


    with open(out_output_filename,"w") as f:
        for token in token_set:
            for state in states:
                prob = output_probs[(token,state)]
                f.writelines(f"{token}\t{state}\t{prob}\n")


# Question 7i. Run Forward-Backward algorithm on cat_price_changes_train, cat_states 
# to generate output probability and transition probability, cat_output_probs.txt and cat_trans_probs.txt
# forward_backward("cat_price_changes_train.txt", "cat_states.txt", "cat_trans_probs.txt", "cat_output_probs.txt",100000,8,1e-4)

def examine_cat_output(output_probs_filename):
    output_probs = read_prob_file(output_probs_filename)
    probs = {}
    probs['s0'] = {}
    probs['s1'] = {}
    probs['s2'] = {}
    semantics = {"x_0_neg_6" : "-6 <= x < 0", "x_0" : "x = 0", "x_0_6" : "0 < x <= 6"}
    for key in probs.keys():
        probs[key]["x_0_6"] = 0
        probs[key]["x_0"] = 0
        probs[key]["x_0_neg_6"] = 0

    for key, value in output_probs.items():
        if key[0] == "-UNK-TOKEN-":
            pass
        else:
            x = int(key[0])
            state = key[1]
            if x <= 6 and x > 0:
                probs[state]["x_0_6"] += value
            elif x == 0:
                probs[state]["x_0"] += value
            elif x >= -6 and x < 0:
                probs[state]["x_0_neg_6"] += value
    
    for key, value in probs.items():
         max_prob = max(value.values())
         max_key = max(value, key = value.get)
         print("state", key, "represents '" + semantics[max_key] + "' which is", max_prob)

# Question 7i:
# State s0 represents the price change is negative from -6 inclusive to 0 exclusive
# State s1 represents the price change is positive from  0 exclusive to 6 inclusive 
# State s2 represents no chance in price, x = 0

# examine_cat_output('cat_output_probs.txt')
### OUTPUT ###
# state s0 represents '-6 <= x < 0' which is 0.9982441337308028
# state s1 represents '0 < x <= 6' which is 0.9352714278873473
# state s2 represents 'x = 0' which is 0.6528617303683221
### END OF OUTPUT ###

def examine_trans_output(trans_output_filename):
    states = {'s0' : 0, 's1': 0, 's2' : 0}
    transitions = {}



def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):
    # output probability
    bj = read_prob_file(in_output_probs_filename)
    # transition probability
    aj_i = read_prob_file(in_trans_probs_filename)
    ai_j = {}
    for key, value in aj_i.items():
        flip = (key[1], key[0])
        ai_j[flip] = aj_i[key]
    # states
    states = read_states_text(in_states_filename)
    # output prediction
    output = []

    # List of list, each list is a sequence of price changes
    test = read_test_text(in_test_filename)
    
    most_likely_outputs = {s:(0,'') for s in states}
    for key,value in bj.items():
        out,s = key
        if most_likely_outputs[s][0] <= value:
            most_likely_outputs[s] = (value,out)

#    print(most_likely_outputs)
    for sequence in test:
        alphas, _ = get_alpha(sequence,ai_j,bj,states)
        final_step_alpha = alphas[-1,:]
        # sum alpha(i,t-1)*ai,j*bj by setting bj to be 1
        next_step_alpha = [sum(final_step_alpha[i] * ai_j[(from_state,to_state)]
                            for i,from_state in enumerate(states))*most_likely_outputs[to_state][0]\
                            for to_state in states]
        state_to_transition = states[np.argmax(next_step_alpha)]
        output.append(most_likely_outputs[state_to_transition][1])
    
    with open(out_predictions_file,"w") as f:
        for price_change in output:
            f.writelines(price_change)
            f.writelines("\n")


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '.' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    # naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    # naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    # naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    # correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    # print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    # naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    # naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    # correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    # print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    # trans_probs_filename =  f'{ddir}/trans_probs.txt'
    # output_probs_filename = f'{ddir}/output_probs.txt'

    # in_tags_filename = f'{ddir}/twitter_tags.txt'
    # viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    # viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
    #                 viterbi_predictions_filename)
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')
    
    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tags_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tags_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    output_probs_filename3 = f'{ddir}/output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    output_probs_filename4 = f'{ddir}/output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}/cat_states.txt'
    out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    out_output_filename = f'{ddir}/cat_output_probs.txt'
    max_iter = 1000000
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    in_states_filename       = f'{ddir}/cat_states.txt'
    predictions_filename     = f'{ddir}/cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                predictions_filename)

    in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    # generate_output_probs_1("twitter_train.txt","twitter_tags.txt","naive_output_probs.txt")
    # generate_output_probs_1("twitter_train.txt","twitter_tags.txt","output_probs.txt")
    # generate_trans_prob("twitter_train.txt","twitter_tags.txt", "trans_probs.txt")
    # generate_output_probs_2("twitter_train.txt","twitter_tags.txt","output_probs2.txt")
    # generate_trans_prob("twitter_train.txt","twitter_tags.txt", "trans_probs2.txt")    
    forward_backward("twitter_train_no_tag.txt", "twitter_tags.txt", "trans_probs3.txt",
                         "output_probs3.txt",0,8,1e-4)
    run()