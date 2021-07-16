from MultiDimDict import MDDict
import sys
import math
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
                    if c3[i] > 0:
                        c1[i][j] = c2[i][j] / c3[i]
                    else:
                        c1[i][j] = 0
            return c1
        else:
            for i in vocab[0]:
                c1[i] = self.divider_helper(c1[i], c2[i], c3[i], vocab[1:])
            return c1


    def calculate_emissions(self):

        axes_for_emissions = []
        axes_for_emissions.append('y')
        axes_for_emissions.append('x')
        vocab_for_emissions = []
        for axe in axes_for_emissions:
            if axe == 'y':
                vocab_for_emissions.append(self.vocab_y)
            elif axe == 'x':
                vocab_for_emissions.append(self.vocab_x)
        self.emissions = MDDict(vocab_for_emissions)
        count_emit = MDDict(vocab_for_emissions)
        count_yi = MDDict(vocab_for_emissions[:-1])

        for i, row in self.data.iterrows():
            _iter = 1
            x = row['x']
            y = row['y']

            while _iter < len(y) - 2:
                _data = [y[_iter]]+[x[_iter-1]] 
                count_emit.add_value(_data,1)
                count_yi.add_value([y[_iter]],1)
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
                vocab_for_transitions.append(self.vocab_y)
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

    def __get_pi(self, p, lst):
        l = lst[0]
        if len(lst) == 1:
            return p[self.vocab_y.index(l)]
        else:
            return self.__get_pi(p[self.vocab_y.index(l)], lst[1:])

    def __calculate_pi(self,pred, iter ,length , lst, list_x, PI):
        if length == 1:
            _ = []
            if iter >= 0:
                for i in self.vocab_y:
                    _lst = [i] + lst
                    a = self.emissions.get_sum([pred]+list_x)
                    b = self.transitions.get_sum(_lst + [pred])
                    nlog_a = sys.maxsize/1000
                    nlog_b = sys.maxsize/1000
                    if a > pow(10,-7):
                        nlog_a = -1 * math.log(a)
                    if b > pow(10,-7):
                        nlog_b = -1 * math.log(b)
                    if len(PI) == 0:
                        _.append(nlog_a+ nlog_b)
                    else:
                        _pi = self.__get_pi(PI[-1], _lst[::-1])
                        _.append(nlog_a+ nlog_b+ _pi)
            else:
                for i in self.vocab_y:
                    _lst = [':'] + lst
                    a = self.emissions.get_sum([pred]+list_x)
                    b = self.transitions.get_sum(_lst + [pred])
                    nlog_a = sys.maxsize/1000
                    nlog_b = sys.maxsize/1000
                    if a > pow(10,-7):
                        nlog_a = -1 * math.log(a)
                    if b > pow(10,-7):
                        nlog_b = -1 * math.log(b)
                    if len(PI) == 0:
                        _.append(nlog_a+ nlog_b)
                    else:
                        _pi = self.__get_pi(PI[-1], _lst[::-1])
                        _.append(nlog_a+ nlog_b+ _pi)
            return min(_)
        else:
            if iter >= 0:
                return [self.__calculate_pi(pred, iter-1, length-1, [i]+lst,list_x, PI) for i in self.vocab_y]
            else: 
                return [self.__calculate_pi(pred, iter-1, length-1, [':']+lst,list_x, PI) for i in self.vocab_y]
    
    def argmin(self, lst):
        return lst.index(min(lst))
    
    def __get_argmax_pi(self, PIn, y_star, len):
        if len == self.num_y:
            _ = []
            for i, v in enumerate(self.vocab_y):
                _.append(self.__get_argmax_pi(PIn[i], y_star, len-1))
            m = self.argmin(_)
            return self.vocab_y[m]
        else:
            sum = 0
            if len == 0:
                return PIn
            else:
                for i,v in enumerate(self.vocab_y):
                    sum += self.__get_argmax_pi(PIn[i], y_star, len-1)
                return sum


    def viterbi(self, test_x):
        ## Final Goal: Find PI(n,v) where n is the length of y without start and stop
        ## We wont use PI we will maximize negative log pi
        ## Step 1: Find PI(1,v)

        k = 1
        PI=[]
        while k < len(test_x)+1:
            list_x = []
            _pi = []
            for i in range(self.num_x):
                if (k - i - 1) >= 0:
                    list_x.append(test_x[k - i - 1])       
                else:
                    list_x.append(':')
            list_x = list_x[::-1]
            
            for pred_y_k in self.vocab_y:
                _pi.append(self.__calculate_pi(pred_y_k, k, self.num_y, [], list_x, PI))

            PI.append(_pi)

            k+= 1
            if k%100 == 0:
                print(k)

        print(len(PI), 'Should be ')



        PI = PI[::-1]
        y_star = 'STOP'
        final_Y = []
        final_Y.append(y_star)
        for index, _PI in enumerate(PI):
            list_x = []
            for i in range(self.num_x):
                if (index - i) >= 0:
                    list_x.append(test_x[index - i])       
                else:
                    list_x.append(':')
            list_x = list_x[::-1]
            y_star = self.__get_argmax_pi(_PI, y_star, self.num_y)
            final_Y.append(y_star)

        final_Y.append('START')
        final_Y = final_Y[::-1]
        return final_Y
     
