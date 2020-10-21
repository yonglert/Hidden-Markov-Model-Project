# helper read text functions
def read_training_text(text_file_name):

    #######
    ###return list of tuples (token, tag)
    #######
    
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

    #######
    ###return list of tokens
    #######
    
    with open(text_file_name) as f:
        data = f.read()
    data = data.split("\n\n")
    if data[-1]=='':
        del data[-1]
    data = [chunk.split("\n") for chunk in data]
    return data

def read_states_text(state_textfile):

    #######
    ###read twitter_tags, return list of tags
    #######
    
    with open(state_textfile) as f:
        data = f.read()
    data = data.split("\n")
    if data[-1]=='':
        del data[-1]
    return data

def read_prob_file(prob_file):

    ########
    ## return dictionary of { (token, state) : p(token | state) } [emission prob]
    ########
    '''
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
'''
Naive prediction 1 approach, generate probs naively by counting y = j -> x = w / count (y = j)
Then use smoothing value of 0.1 to smooth and ensure no zero probability.
i.e prob(token | state) = count(y=j -> x=w)
'''
from collections import Counter
smooth_value = [0.01, 0.1, 1, 10]
def generate_output_probs_1(training_file, states_file, 
                           output_prob_file, smooth_value = smooth_value[1]):
    #generate naively p(x=w|y=j) as
    #count(y = j -> x = w)/count(y = j)
    #outputs to format of x\ty\tp(x = w|y = j) for each row

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
    else:
        return word

import re
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
    token_set.add("-UNK-TOKEN-") #unknown
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

'''
Using naively generated bj(w) which is P(x = w | y = j)
we find best tag j* which is argmax bj(w) = argmax P(x = w |y =j)

Accuracy -> 64.8%
'''
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
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

naive_predict("naive_output_probs.txt","twitter_dev_no_tag.txt","naive_prediction.txt")

'''
Naive prediction 2 approach: argmax P(y = j | x = w) to find j*
i.e given the word, what is the most suitable tag for the word?

to get P(y = j | x = w), we find using P(x = w | y = j) * p(y = j) / p(x = w) aka Bayes Rule
p(y = j) -> count(y = j) / total count of Y
p(x = w) -> count(x = w) / total count of X

then find argmax...
Accuracy -> 69.4% 
'''

def naive_predict2(in_output_probs_filename,in_train_file_name, in_test_filename, out_prediction_filename, smooth_value = 0.1):
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
    smooth_value_denom_x = (n_words + 1) * smooth_value
    smooth_value_denom_y = (n_states + 1) * smooth_value
    ####################################

    with open(out_prediction_filename,"w") as f:
        for chunk in test_data:
            for token in chunk:
                token = token if token in known_tokens else "-UNK-TOKEN-"
                num_x = word_count[token] + 1
                p_x = (num_x * smooth_value) / smooth_value_denom_x
                token_probs = sorted([[(p_x_given_y[(token,state)] * \
                    (((state_count[state]+1)*smooth_value)/smooth_value_denom_y)) / p_x ,state] for state in states],key=lambda x:x[0])
                max_prob_state = token_probs[-1][1]
                f.writelines(f"{max_prob_state}\n")
            f.writelines(f"\n") #extra newline per tweet

naive_predict2("naive_output_probs.txt", 'twitter_train.txt','twitter_dev_no_tag.txt', 'naive_predictions2.txt')

from itertools import permutations
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
                


import re
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    states = read_states_text(in_tags_filename)
    trans_probs = read_prob_file(in_trans_probs_filename)
    emission_probs = read_prob_file(in_output_probs_filename)
    test_data = read_test_text(in_test_filename)
    known_tokens = set((key[0] for key in emission_probs.keys()))

    # for i in range(len(test_data)):
    #     if test_data[i] in known_tokens:
    #         continue
    #     elif (re.search('^@',test_data[i]) != None):
    #         test_data[i] = "-@mention-"
    #     elif (re.search('^#',test_data[i]) != None):
    #         test_data[i] = "-#hashtag-"
    #     elif (re.search('(http|ftp|https)(:\/\/)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',test_data[i]) != None):
    #         test_data[i] = "-httplink-"
    #     else:
    #         test_data[i] = "-UNK-TOKEN-"

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

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    pass

def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):
    pass


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

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    pass

    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')
    
    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
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
    generate_output_probs_1("twitter_train.txt","twitter_tags.txt","naive_output_probs.txt")
    generate_output_probs_1("twitter_train.txt","twitter_tags.txt","output_probs.txt")
    generate_trans_prob("twitter_train.txt","twitter_tags.txt", "trans_probs.txt")
    generate_output_probs_2("twitter_train.txt","twitter_tags.txt","output_probs2.txt")
    generate_trans_prob("twitter_train.txt","twitter_tags.txt", "trans_probs2.txt")    
    run()