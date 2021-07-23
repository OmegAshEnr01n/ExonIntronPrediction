# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Drosophila Melanogaster

# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import Bio
import torch as torch
from torch.utils.data import Dataset, DataLoader
print("Biopython v" + Bio.__version__)
from timeit import default_timer as timer
import copy
import math
import sys



# %%
from Bio import SeqIO
count = 0
sequences = [] # Here we are setting up an array to save our sequences for the next step

for seq_record in SeqIO.parse("./Genomedata/genome.fa", "fasta"):
    if (count < 6):
        sequences.append(seq_record)
        print("Id: " + seq_record.id + " \t " + "Length: " + str("{:,d}".format(len(seq_record))) )
        print(repr(seq_record.seq) + "\n")
        count = count + 1

sequences.pop()


# %%
sequences


# %%
agsts = pd.read_csv('./Genomedata/genes-augustus.csv')


# %%
agsts['chrom'].unique()

# %% [markdown]
# Finding the sequences 

# %%
def code_Exon_Intron(coding_row, ch):
    ch[coding_row['txStart']:coding_row['txEnd']] = 'T'
    Estart = np.array(coding_row['exonStarts'].split(',')[:-1]).astype(int)
    Eend = np.array(coding_row['exonEnds'].split(',')[:-1]).astype(int)
    for i in range(len(Estart)):
        ch[Estart[i]:Eend[i]+1] = 'E'
        if i != len(Estart) - 1:
            ch[Eend[i]+1:Estart[i+1]] = 'I'
    return ch

def code_sequence(augustus, sequences):
    df = pd.DataFrame(columns=['id','x','y'])
    
    for seq in sequences:
        s_id_aug = augustus.loc[augustus['chrom'] == seq.id]

        y = np.array(['N'] * s_id_aug.iloc[-1]['txEnd'],dtype='U')

        t = np.array([rec for rec in str(seq.seq)],dtype='U')[:s_id_aug.iloc[-1]['txEnd']]
        x = np.array([x.upper() if isinstance(x, str) else x for x in t])


        for index, row in s_id_aug.iterrows():

            if row['name'][-3:] == '.t2':
                y = code_Exon_Intron(row, y)
                
            check_t2 = row['name'][:-3] + '.t2'
            if s_id_aug[s_id_aug['name'].isin([check_t2])].empty:
                continue
            else: 
                y = code_Exon_Intron(row, y)
        # print(x[0],'\n',x[1],'\n/',x[2],'\n',x[3],'\n',x[100])
        print(len(x.tolist()))
        char = pd.Series({'id': seq.id, 'x':x.tolist(), 'y':y.tolist()})
        print(char['x'][:20])
        df = df.append(char,ignore_index=True)
    return df


# %%
temp = code_sequence(agsts, sequences)


# %%
def prepare_segment(x,y):
    df = pd.DataFrame(columns = ['x','y'])
    last_i = 0
    for i in range(1000,len(x),1000):
        df = df.append({'x': x[last_i:i], 'y':['START'] + y[last_i:i] + ['STOP']}, ignore_index = True)
        last_i = i
    return df


# %%
df = prepare_segment(temp.iloc[4]['x'],temp.iloc[4]['y'])


# %%
class MultiDimnDict():
    
    def __init__(self,axes):
        self.axes = axes

        self.data = {}
        prev = None

        for i in range(len(axes)-1, -1, -1):
            x = {}
            for n in axes[i]:
                if prev == None:
                    x[n] = 0
                else:
                    x[n] = copy.deepcopy(prev)
            prev = x
            self.data = x

    def __get_string(self, x,_str, depth):

        if type(list(x.values())[0]) == type(dict()):
            
            for i in x:

                _str += self.__get_string(x[i], depth * " " + i + "  \n", depth+1)
                # print(_str)
            return _str
        else:
            return _str + " " * depth +str(x) +"\n"

    def __str__(self):
        string = self.__get_string(self.data, "", 0)
        return string

    def add_value_helper(self,lst,val, temp, depth):
        label = lst[0]
        if len(lst) == 1:
            if label == ':':
                for x in self.axes[depth]:
                    temp[x] += val
            else:
                temp[label] += val
            return temp
        
        # print(lst[1:],val,temp[lst[0]])
        if label == ':':
            for x in self.axes[depth]:
                temp[x] = self.add_value_helper(lst[1:], val, temp[x], depth+1)
        else:
            temp[label] = self.add_value_helper(lst[1:],val,temp[label], depth+1)
        return temp


    def add_value(self,lst,val):
        if len(lst) == len(self.axes):
            self.data =  self.add_value_helper(lst,val,self.data, 0)
            return self
        else:
            return 0
    
    def get_sum(self, lst):
        sum = self.get_sum_helper(lst,self.data ,0)
        return sum

    def get_sum_helper(self,lst, temp, depth):
        index = lst[0]
        if len(lst) == 1:
            return temp[index]
        else:
            if index == ':':
                sum = 0
                for x in self.axes[depth]:
                    sum += self.get_sum_helper(lst[1:], temp[x], depth+1)
                return sum
            else:
                return self.get_sum_helper(lst[1:], temp[index], depth+1)
    

x= MultiDimnDict([['A', 'N', 'G', 'C', 'T'], ['E', 'START', 'STOP', 'N', 'I'], ['E', 'START', 'STOP', 'N', 'I']])

print(x.add_value(['A','N','N'], 1))

# %% [markdown]
# Cant write to dataframe since the character array turns into shit
# %% [markdown]
# Steps
# 
# 1. Create the count matrices
# 2. Use viterbi algo


# %%
class HMM():
    def __init__(self,data,num_x,num_y,vocab_x,vocab_y):
        self.data = data
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.num_x = num_x
        self.num_y = num_y
    
    def helper(self, c1,c2,c3, vocab):
        if len(vocab) == 1:
            for i in vocab[0]:
                if c3[i] > 0:
                    c1[i] = c2[i] / c3[i]
                else:
                    c1[i] = 0
            return c1
        else:
            for i in vocab[0]:
                c1[i] = self.helper(c1[i], c2[i], c3, vocab[1:])
            return c1


    def calculate_emissions(self):

        axes_for_emissions = []
        for i in range(self.num_x):
            axes_for_emissions.append('x')
        for i in range(self.num_y):
            axes_for_emissions.append('y')
        vocab_for_emissions = []
        for axe in axes_for_emissions:
            if axe == 'y':
                vocab_for_emissions.append(list(self.vocab_y))
            elif axe == 'x':
                vocab_for_emissions.append(list(self.vocab_x))
        self.emissions = MultiDimnDict(vocab_for_emissions)
        count_emit = MultiDimnDict(vocab_for_emissions)
        count_yi = MultiDimnDict([list(self.vocab_y)])
        # count_transition_last = MultiDimnDict([list(self.vocab_y),['STOP']])
        # self.transition_last = MultiDimnDict([list(self.vocab_y),['STOP']])

        for i, row in self.data.iterrows():
            _iter = 1
            x = row['x']
            y = row['y']

            while _iter < len(y) - 2:
                _data = []
                list_y = []
                for i in range(self.num_y):
                    if (_iter - i) >= 0:
                        list_y.append(y[_iter - i])
                    else:
                        list_y.append(':')
                list_y = list_y[::-1] # so that the list is []....,y-2, y-1, y]
                list_x = []
                for i in range(self.num_x):
                    if (_iter - i - 1) >= 0:
                        list_x.append(x[_iter - i - 1])       
                    else:
                        list_x.append(':')
                list_x = list_x[::-1]
                _data = list_x + list_y
                count_emit.add_value(_data,1)
                count_yi.add_value([list_y[-1]],1)

                _iter += 1
            
            # count_transition_last.add_value([y[_iter-1],'STOP'], 1)
            # count_yi.add_value(['STOP'], 1)

        

            
        self.emissions.data = self.helper(self.emissions.data, count_emit.data, count_yi.data, vocab_for_emissions)
        # self.transition_last.data = self.helper(self.transition_last.data, count_transition_last.data, count_yi.data, [list(self.vocab_y),['STOP']])
        return self.emissions




    def viterbi(self, test_x):
        ## Final Goal: Find PI(n,v) where n is the length of y without start and stop
        ## We wont use PI we will maximize negative log pi
        ## Step 1: Find PI(1,v)

        k = 1
        viterbi_y = ['START']
        PI=[]
        _pi = []
        while k < len(test_x)+1:
            list_x = []
            list_y = []
            for i in range(self.num_x):
                if (k - i - 1) >= 0:
                    list_x.append(test_x[k - i - 1])       
                else:
                    list_x.append(':')
            list_x = list_x[::-1]
            for i in range(self.num_y):
                if i == 0:
                    continue
                elif (k-i) >= 0:
                    list_y.append(viterbi_y[k-i])
                else:
                    list_y.append(':')
            list_y = list_y[::-1]

            _pi = []
            for pred_y_k in self.vocab_y:
                _data = list_x + list_y + [pred_y_k]
                a = self.emissions.get_sum(_data)
                nlog_a = sys.maxsize
                if a > pow(10,-7):
                    nlog_a = -1 * math.log(a)
                if len(PI) == 0:
                    _pi.append(nlog_a)
                else: 
                    
                    _pi.append(nlog_a + PI[-1][list(self.vocab_y).index(viterbi_y[-1])])

            min_pi = min(_pi)
            min_y = list(self.vocab_y)[_pi.index(min_pi)]

            viterbi_y.append(min_y)
            PI.append(_pi)

            k+= 1
        print(len(PI), 'Should be 1000')
        print(len(viterbi_y), 'Should be 1001 including Start tag')


        PI = PI[::-1]
        y_star = 'STOP'
        final_Y = []
        final_Y.append(y_star)
        for i in range(self.num_y -1):
            final_Y.append(viterbi_y[len(viterbi_y)-1-i])
            del PI[0]
        for _PI in PI:
            pred_pi = []
            index = PI.index(_PI)
            list_x = []
            list_y = []
            for i in range(self.num_x):
                if (index - i) >= 0:
                    list_x.append(test_x[index - i])       
                else:
                    list_x.append(':')
            list_x = list_x[::-1]
            for i in range(self.num_y-1):
                if (index-i) >= 0:
                    list_y.append(viterbi_y[index-i])
                else:
                    list_y.append(':')
            list_y = list_y[::-1]
            for y in self.vocab_y:
                _data = list_x + [y] + list_y 
                emission = sys.maxsize
                a = self.emissions.get_sum(_data)
                if a > pow(10,-7):
                    emission = math.log(a) * -1
                pred_pi.append(_PI[list(self.vocab_y).index(y)] + emission)


            min_pi = min(pred_pi)
            final_y = list(self.vocab_y)[pred_pi.index(min_pi)]


            final_Y.append(final_y)

        final_Y.append('START')
        final_Y = final_Y[::-1]
        viterbi_y += ['STOP']
        return final_Y, viterbi_y
            


        # temp_y = []
        # _pi = []
        # _y = []
        # PI = []
        # k = 0
        # x = test_x[k]
        # for y in self.vocab_y:
        #     a = self.emissions_yi_xi.data[y][x]
        #     b = self.transitions.data['START'][y]


        #     nlog_a = sys.maxsize 
        #     nlog_b = sys.maxsize 
        #     if a > pow(10,-7):
        #         nlog_a = -1 * math.log(a)
        #     if b > pow(10,-7):
        #         nlog_b = -1 * math.log(b)
        #     _pi.append(nlog_a)
        #     _y.append(y)
            
        # min_pi = min(_pi)
        # min_y = _y[_pi.index(min_pi)]

        # yi1 = min_y
        # PI.append(min_pi)
        # temp_y.append(yi1)

        # print("Done 1 ", _pi)

        # k = 1
        # while k < len(test_x)-1:
        #     x = test_x[k]
        #     xi = test_x[k-1]
        #     _pi = []
        #     _y = []
        #     for y in self.vocab_y:
        #         a = self.emissions.data[y][x][xi]
        #         b = self.transitions.data[yi1][y]

        #         nlog_a = sys.maxsize 
        #         nlog_b = sys.maxsize 
        #         if a > pow(10,-7):
        #             nlog_a = -1 * math.log(a)
        #         if b > pow(10,-7):
        #             nlog_b = -1 * math.log(b)
                
        #         _pi.append(nlog_a + PI[-1])

        #         _y.append(y)
                

        #     min_pi = min(_pi)
        #     min_y = _y[_pi.index(min_pi)]

        #     yi1 = min_y
        #     PI.append(min_pi)
        #     temp_y.append(yi1)
        #     k += 1 

        #     if k % 100000 == 0:
        #         print("k = ",k,"    pi = ",PI[-5:])
        
        # print('PI= ',len(PI))
        # input()
        # y_star = 'STOP'
        # final_Y = []
        # final_Y.append(y_star)


        # for _PI in PI[::-1]:
        #     _pi = []
        #     _y = []
        #     index = PI.index(_PI)
        #     xi1 = test_x[index -1]
        #     xi = test_x[index]
        #     for y in self.vocab_y:
        #         emission = sys.maxsize
        #         a = self.emissions.data[y][xi1][xi]
        #         if a > pow(10,-7):
        #             emission = math.log(a) * -1
        #         _pi.append(_PI + emission)
        #         _y.append(y)

        #     min_pi = min(_pi)
        #     final_y = _y[_pi.index(min_pi)]

        #     if index % 5000 == 0:
        #         print(index,"   ",_pi)
        #     # print('\n\n','final_y ',final_y,'\nxi= ',xi)

        #     final_Y.append(final_y)

        
        # _pi = []
        # _y = []
        # xi = test_x[0]
        # for y in self.vocab_y:
        #     emission = sys.maxsize
        #     a = self.emissions_yi_xi.data[y][xi]
        #     if a > pow(10,-7):
        #         emission = math.log(a) * -1
        #     _pi.append(_PI + emission)
        #     _y.append(y)

        # min_pi = min(_pi)
        # final_y = _y[_pi.index(min_pi)]   

        # final_Y.append(final_y)   

        # final_Y.append('START')
        # return final_Y[::-1],temp_y


# %%
hmm = HMM(df,3,6 ,set(temp.iloc[0]['x']), set(temp.iloc[0]['y'] + ['START','STOP']))


# %%
start = timer()
hmm.calculate_emissions()
stop = timer()
print(stop-start)


# %%
start = timer()
y_pred, vy = hmm.viterbi(temp.iloc[4]['x'])
stop = timer()
print(stop-start)


# %%
print(len(y_pred),'\n',len(temp.iloc[4]['y']))
print(y_pred[-1], temp.iloc[4]['y'][-1])


# %%
print(y_pred)


# %%
count = 0
y_corr = ['START'] + temp.iloc[4]['y'] + ['STOP']
for i in range(len(y_pred)):
    if y_pred[i] == y_corr[i]:
        count += 1
print(count/len(y_pred))


# %%

print(set(y_corr))
print(set(temp_y))
print(set(y_pred))
print([y_corr.count(i) for i in set(y_corr)])
print([temp_y.count(i) for i in set(temp_y)])
print([y_pred.count(i) for i in set(y_pred)])


# %%
start = timer()
y_pred,temp_y = hmm.viterbi(temp.iloc[4]['x'][:20000])
stop = timer()
print(stop-start)


# %%
print(len(y_pred),'\n',len(temp.iloc[4]['y'][:20000]))
print(y_pred[-1], temp.iloc[4]['y'][-1])


# %%
count = 0
y_corr = ['START'] + temp.iloc[4]['y'][:20000] + ['STOP']
print(len(y_pred), len(y_corr))
for i in range(len(y_pred)):
    if y_pred[i] == y_corr[i]:
        count += 1
print(count/len(y_pred))


# %%

print(set(y_corr))
print(set(temp_y))
print(set(y_pred))
print([y_corr.count(i) for i in set(y_corr)])
print([temp_y.count(i) for i in set(temp_y)])
print([y_pred.count(i) for i in set(y_pred)])


# %%



# %%



# %%



# %%



