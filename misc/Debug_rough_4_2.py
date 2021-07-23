# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]

So I have stopped this because I have run into the problem of the fcat that my viterbi algo and emissions were no n order.
They were some creation of my own. 

Step 1: creat n order library
Step 2: create modification for emissions to contain ys and xs. pi values will only be updated on y
Step 3: move to other models

# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import Bio
print("Biopython v" + Bio.__version__)
from timeit import default_timer as timer
from gtfparse import read_gtf
import copy
import math
import sys


# %% [markdown]


class MDDict():
    
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
    
class HMM():
    def __init__(self,data,num_y,vocab_x,vocab_y):
        self.data = data
        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.num_y = num_y 
        self.num_x = 1
    
    def divider_helper(self, c1,c2,c3, vocab):
        if len(vocab) == 2:
            for i in vocab[0]:
                for j in vocab[1]:
                    if c3[j] > 0:
                        c1[i][j] = c2[i][j] / c3[j]
                    else:
                        c1[i][j] = 0
            return c1
        else:
            for i in vocab[0]:
                c1[i] = self.divider_helper(c1[i], c2[i], c3[i], vocab[1:])
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
        self.emissions = MDDict(vocab_for_emissions)
        count_emit = MDDict(vocab_for_emissions)
        count_yi = MDDict(vocab_for_emissions[1:])

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
                count_yi.add_value(list_y,1)

                _iter += 1
            

            
        self.emissions.data = self.divider_helper(self.emissions.data, count_emit.data, count_yi.data, vocab_for_emissions)
        return self.emissions


    def calculate_transitions(self):

        axes_for_transitions = []
        for i in range(self.num_y + 1):
            axes_for_transitions.append('y')
        vocab_for_transitions = []
        for axe in axes_for_transitions:
            if axe == 'y':
                vocab_for_transitions.append(list(self.vocab_y))
        self.transitions = MDDict(vocab_for_transitions)
        count_transit = MDDict(vocab_for_transitions)
        count_yi = MDDict(vocab_for_transitions[1:])

        for i, row in self.data.iterrows():
            _iter = 1
            y = row['y']

            while _iter < len(y) - 2:
                _data = []
                list_y = []
                for i in range(self.num_y+1):
                    if (_iter - i) >= 0:
                        list_y.append(y[_iter - i])
                    else:
                        list_y.append(':')
                list_y = list_y[::-1] # so that the list is []....,y-2, y-1, y]

                _data = list_y
                count_transit.add_value(_data,1)
                count_yi.add_value(list_y[:-1],1)

                _iter += 1
            

            
        self.transitions.data = self.divider_helper(self.transitions.data, count_transit.data, count_yi.data, vocab_for_transitions)
        return self.transitions
 
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
            for i in range(self.num_y+1):
                if i == 0:
                    continue
                elif (k-i) >= 0:
                    list_y.append(viterbi_y[k-i])
                else:
                    list_y.append(':')
            list_y = list_y[::-1]

            _pi = []

            for pred_y_k in self.vocab_y:
                _data = list_x + list_y[1:] + [pred_y_k]
                a = self.emissions.get_sum(_data)
                b = self.transitions.get_sum(list_y + [pred_y_k])
                nlog_a = sys.maxsize
                nlog_b = sys.maxsize
                if a > pow(10,-7):
                    nlog_a = -1 * math.log(a)
                if b > pow(10,-7):
                    nlog_b = -1 * math.log(b)
                if len(PI) == 0:
                    _pi.append(nlog_a+nlog_b)
                else: 
                    _pi.append(nlog_a + nlog_b+ PI[-1][self.vocab_y.index(viterbi_y[-1])])

            min_pi = min(_pi)
            min_y = list(self.vocab_y)[_pi.index(min_pi)]

            viterbi_y.append(min_y)
            PI.append(_pi)

            k+= 1
        print(len(PI), 'Should be 1000')
        print(len(viterbi_y), 'Should be 1001 including Start tag')

        viterbi_y = viterbi_y[::-1]
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
            for i in range(self.num_y):
                if (index-i) >= 0:
                    list_y.append(viterbi_y[index-i])
                else:
                    list_y.append(':')
            list_y = list_y[::-1]
            for y in self.vocab_y:
                _data = list_x + [y] + list_y 
                emission = sys.maxsize
                transition = sys.maxsize
                a = self.emissions.get_sum(_data)
                if a > pow(10,-7):
                    emission = math.log(a) * -1
                if b > pow(10,-7):
                    transition = -1 * math.log(b)
                pred_pi.append(_PI[list(self.vocab_y).index(y)] + emission + transition)


            min_pi = min(pred_pi)
            final_y = list(self.vocab_y)[pred_pi.index(min_pi)]


            final_Y.append(final_y)

        final_Y.append('START')
        final_Y = final_Y[::-1]
        viterbi_y += ['STOP']
        return final_Y, viterbi_y
     


# %%
from Bio import SeqIO
count = 0
sequences = [] # Here we are setting up an array to save our sequences for the next step

for seq_record in SeqIO.parse("./Panthera_leo/Panthera_leo.PanLeo1.0.dna.primary_assembly.A1.fa", "fasta"):
    if (count < 6):
        sequences.append(seq_record)
        print("Id: " + seq_record.id + " \t " + "Length: " + str("{:,d}".format(len(seq_record))) )
        print(repr(seq_record.seq) + "\n")
        count = count + 1


# %%
sequences


# %%
gtf = read_gtf('./Panthera_leo/Panthera_leo.PanLeo1.0.104.gtf')


# %%
gtf.head(10)


# %%
gtf['seqname'].unique()


# %%
gtfi = gtf.drop(['source','gene_source','transcript_id','transcript_version','transcript_name','gene_name','projection_parent_transcript','gene_version', 'gene_id','transcript_source','score'],axis = 1)


# %%
gtf.loc[gtf['seqname'] == 'A1'].iloc[7]


# %%
gtfi_a1 = gtfi.loc[gtfi['seqname'] == 'A1']


# %%
def check_single_exon(i, Y):
    y = Y.iloc[i]
    count_exon = 0
    while y['feature'] != 'gene':
        y = Y.iloc[i]
        if y['feature'] == 'exon':
            count_exon += 1
        i += 1
    return count_exon == 1

def code_single_exon(start, end, is_reverse):
    y = []
    num = int((end-start+1)/3)
    numr = (end-start+1)%3
    if is_reverse:
        y += ['_E1', '_E2', '_E3'] * num
        if numr == 1:
            y+=['_E1']
        elif numr == 2:
            y+=['_E1', '_E2']
    else:
        y += ['E1', 'E2', 'E3'] * num
        if numr == 1:
            y+=['E1']
        elif numr == 2:
            y+=['E1', 'E2']
    return y


def code_exon(start, end, is_reverse, X, last_exon):
    y = []
    g = X.count('G')
    c = X.count('C')
    a = X.count('A')
    t = X.count('T')
    GC = True if (g+c)/(a+t+g+c) > 0.5 else False
    if is_reverse:
        if GC:
            num = int((end - start + 1)/3)
            numr = (end-start+1)%3
            if last_exon == '3':
                y = ['_EH1','_EH2','_EH3'] * num
            elif last_exon == '2':
                y = ['_EH3','_EH1','_EH2'] * num
            elif last_exon == '1':
                y = ['_EH2','_EH3','_EH1'] * num

            if num == 0:
                if last_exon == '1':
                    if numr == 2:
                        y += ['_EH2', '_EH3']
                    elif numr == 1:
                        y += ['_EH2']
                elif last_exon == '2':
                    if numr == 2:
                        y += ['_EH3', '_EH1']
                    elif numr == 1:
                        y += ['_EH3']
                elif last_exon == '3':
                    if numr == 2:
                        y += ['_EH1', '_EH2']
                    elif numr == 1:
                        y += ['_EH1']

            else:
                if y[-1] == '_EH1':
                    if numr == 2:
                        y += ['_EH2', '_EH3']
                    elif numr == 1:
                        y += ['_EH2']
                elif y[-1] == '_EH2':
                    if numr == 2:
                        y += ['_EH3', '_EH1']
                    elif numr == 1:
                        y += ['_EH3']
                elif y[-1] == '_EH3':
                    if numr == 2:
                        y += ['_EH1', '_EH2']
                    elif numr == 1:
                        y += ['_EH1']
        else:
            num = int((end - start + 1)/3)
            numr = (end - start+1)%3
            if last_exon == '3':
                y = ['_EL1','_EL2','_EL3'] * num
            elif last_exon == '2':
                y = ['_EL3','_EL1','_EL2'] * num
            elif last_exon == '1':
                y = ['_EL2','_EL3','_EL1'] * num


            if num == 0:
                if last_exon == '1':
                    if numr == 2:
                        y += ['_EL2', '_EL3']
                    elif numr == 1:
                        y += ['_EL2']
                elif last_exon == '2':
                    if numr == 2:
                        y += ['_EL3', '_EL1']
                    elif numr == 1:
                        y += ['_EL3']
                elif last_exon == '3':
                    if numr == 2:
                        y += ['_EL1', '_EL2']
                    elif numr == 1:
                        y += ['_EL1']
            else:
                if y[-1] == '_EL1':
                    if numr == 2:
                        y += ['_EL2', '_EL3']
                    elif numr == 1:
                        y += ['_EL2']
                elif y[-1] == '_EL2':
                    if numr == 2:
                        y += ['_EL3', '_EL1']
                    elif numr == 1:
                        y += ['_EL3']
                elif y[-1] == '_EL3':
                    if numr == 2:
                        y += ['_EL1', '_EL2']
                    elif numr == 1:
                        y += ['_EL1']
        
    else:
        if GC:
            num = int((end - start + 1)/3)
            numr = (end - start+1)%3
            if last_exon == '3':
                y = ['EH1','EH2','EH3'] * num
            elif last_exon == '2':
                y = ['EH3','EH1','EH2'] * num
            elif last_exon == '1':
                y = ['EH2','EH3','EH1'] * num
            
            if num == 0:
                if last_exon == '1':
                    if numr == 2:
                        y += ['EH2', 'EH3']
                    elif numr == 1:
                        y += ['EH2']
                elif last_exon == '2':
                    if numr == 2:
                        y += ['EH3', 'EH1']
                    elif numr == 1:
                        y += ['EH3']
                elif last_exon == '3':
                    if numr == 2:
                        y += ['EH1', 'EH2']
                    elif numr == 1:
                        y += ['EH1']
            else:
                if y[-1] == 'EH1':
                    if numr == 2:
                        y += ['EH2', 'EH3']
                    elif numr == 1:
                        y += ['EH2']
                elif y[-1] == 'EH2':
                    if numr == 2:
                        y += ['EH3', 'EH1']
                    elif numr == 1:
                        y += ['EH3']
                elif y[-1] == 'EH3':
                    if numr == 2:
                        y += ['EH1', 'EH2']
                    elif numr == 1:
                        y += ['EH1']
        else:
            num = int((end - start + 1)/3)
            numr = (end - start+1)%3
            if last_exon == '3':
                y = ['EL1','EL2','EL3'] * num
            elif last_exon == '2':
                y = ['EL3','EL1','EL2'] * num
            elif last_exon == '1':
                y = ['EL2','EL3','EL1'] * num
            if num == 0:
                if last_exon == '1':
                    if numr == 2:
                        y += ['EL2', 'EL3']
                    elif numr == 1:
                        y += ['EL2']
                elif last_exon == '2':
                    if numr == 2:
                        y += ['EL3', 'EL1']
                    elif numr == 1:
                        y += ['EL3']
                elif last_exon == '3':
                    if numr == 2:
                        y += ['EL1', 'EL2']
                    elif numr == 1:
                        y += ['EL1']
            else:
                if y[-1] == 'EL1':
                    if numr == 2:
                        y += ['EL2', 'EL3']
                    elif numr == 1:
                        y += ['EL2']
                elif y[-1] == 'EL2':
                    if numr == 2:
                        y += ['EL3', 'EL1']
                    elif numr == 1:
                        y += ['EL3']
                elif y[-1] == 'EL3':
                    if numr == 2:
                        y += ['EL1', 'EL2']
                    elif numr == 1:
                        y += ['EL1']
        last_exon = y[-1][-1]
    return last_exon, y

def code_intron(start, end, is_reverse):
    y = []
    if end > start:
        end = end - 1
    else:
        end = end + 1
    length = abs(end - start) - 4
    if length < 0:
        num = int(abs(end - start)/3)
        numr = abs(end - start)%3
        if is_reverse:
            y += ['_I1', '_I2', '_I3'] * num
            if numr == 1:
                y+=['_I1']
            elif numr == 2:
                y+=['_I1', '_I2']
        else:
            y += ['ID2', 'ID3', 'ID1'] * num
            if numr == 1:
                y+=['ID2']
            elif numr == 2:
                y+=['ID2', 'ID3']
        return y

    if is_reverse:
        if length > 100:
            y += ['_Id1','_Id2','_Id3']*33
            y += ['_Id1']
            num = int((length-100)/3)
            numr = (length - 100)%3
            y += ['_ID2', '_ID3', '_ID1'] * num
            if numr == 1:
                y+=['_ID2']
            elif numr == 2:
                y+=['_ID2', '_ID3']

        else:
            num = int(length/3)
            numr = length%3
            y += ['_I1', '_I2', '_I3'] * num
            if numr == 1:
                y+=['_I1']
            elif numr == 2:
                y+=['_I1', '_I2']
        y = ['_ASS', '_ASS'] + y + ['_DSS','_DSS']
    else:
        if length > 100:
            y += ['Id1','Id2','Id3']*33
            y += ['Id1']
            num = int((length-100)/3)
            numr = (length - 100)%3
            y += ['ID2', 'ID3', 'ID1'] * num
            if numr == 1:
                y+=['ID2']
            elif numr == 2:
                y+=['ID2', 'ID3']

        else:
            num = int(length/3)
            numr = length %3
            y += ['I1', 'I2', 'I3'] * num
            if numr == 1:
                y+=['I1']
            elif numr == 2:
                y+=['I1', 'I2']
        y = ['ASS', 'ASS'] + y + ['DSS','DSS']

    return y


def code_gene(i , X , Y, is_reverse,strand, single_exon):
    gene_x = []
    gene_y = []
    y = Y.iloc[i]
    if y['feature'] == 'transcript':
        start = int(y['start'])
        end = int(y['end'])
        if strand:
            gene_x = X[0][start-1:end]
        else:
            gene_x = X[1][start-1:end]
    else:
        print('error')
    first_exon = True
    last_exon = '3'
    end_exon = 0
    while y['feature'] != 'gene':
        y = Y.iloc[i]
        if y['feature'] == 'exon' and first_exon:
            start = int(y['start'])
            end = int(y['end'])
            if is_reverse:
                end_exon = start
            else:
                end_exon = end

            if single_exon:
                gene_y += code_single_exon(start, end, is_reverse)
            else:
                last_exon, lst = code_exon(start, end, is_reverse ,gene_x, last_exon)
                gene_y += lst
            first_exon = False
            
        elif y['feature'] == 'exon' and not first_exon:
            start = int(y['start'])
            end = int(y['end'])
            if is_reverse:
                lst = code_intron(end_exon, end, is_reverse) # the index of end_exon and start is not included. 
            else:
                lst = code_intron(end_exon, start, is_reverse)
            if is_reverse:
                end_exon = start
            else:
                end_exon = end
            gene_y += lst
            last_exon, lst = code_exon(start, end, is_reverse ,gene_x, last_exon)
            gene_y += lst

        i += 1

        

    return i, gene_x, gene_y

def check_gene(i, Y):
    y = Y.iloc[i]
    start_codon = 0
    stop_codon = 0
    strand = y['strand'] == '+'
    while y['feature'] != 'gene':
        y = Y.iloc[i]
        if y['feature'] == 'start_codon':
            start_codon = int(y['start'])
        elif y['feature'] == 'stop_codon':
            stop_codon = int(y['start'])

        i+=1
        

    return start_codon > stop_codon, strand # returns True if the strand is positive and doesnt need ot be complemented

def create_dataset(X, Y):
    i = 0
    count = 0
    df = pd.DataFrame(columns = ['x','y'])
    gene_x = []
    vocab_x = []
    vocab_y = []
    gene_x.append(np.array([rec for rec in str(X.upper())],dtype='U').tolist())
    gene_x.append(np.array([rec for rec in str(X.complement().upper())],dtype='U').tolist())
    while i < len(Y):
        y = Y.iloc[i]
        if y['gene_biotype'] == 'protein_coding' and y['feature'] == 'gene':
            is_reverse, strand = check_gene(i+1, Y)
            single_exon = check_single_exon(i,Y)
            i,__x, __y = code_gene(i+1,gene_x, Y, is_reverse, strand, single_exon)
            # if __x[:3] != ['A','T','G'] and __x[:3] != ['A','G','T'] and __x[:3] != ['A','A','T'] and __x[:3] != ['G','A','T']:
            #     print(len(df))
            # print(len(__x), '   ', len(__y))
            __y = ['START'] + __y + ['STOP']
            vocab_y = vocab_y + list(set(__y))
            vocab_y = list(set(vocab_y))
            vocab_x = vocab_x + list(set(__x))
            vocab_x = list(set(vocab_x))
            if (len(__x)+2) == len(__y):
                datapoint = pd.Series({'x': __x, 'y':__y})
                df = df.append(datapoint, ignore_index=True)
            else:
                count += 1
        else:
            i+=1
    print(count, ' is count failed')
    return df, vocab_x, vocab_y


# %%
start = timer()
df, vocab_x, vocab_y = create_dataset(sequences[0].seq, gtfi_a1)
stop = timer()
print(stop - start)

# %%
print(vocab_y)
# %%
hmm = HMM(df, 1, vocab_x, vocab_y)

# %%
start = timer()
hmm.calculate_emissions()
stop = timer()
print(stop-start)

# %%
start = timer()
hmm.calculate_transitions()
stop = timer()
print(stop-start)
# %%
test_x = df.iloc[0]['x']
y, _y = hmm.viterbi(test_x)
# %%
