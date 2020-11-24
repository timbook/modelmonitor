import numpy as np
import pandas as pd

class ModelMonitor:
    def __init__(self, metric, labels=None, subset=None, sep='_'):
        self.metric = metric
        self.labels = labels
        self.subset = subset
        self.sep = sep
        
    def evaluate_1dim_pair(self, x1, x2):
        return self.metric(x1, x2)
    
    def evaluate_2dim_pair(self, x1, x2):
        assert x1.shape[1] == x2.shape[1], \
            "Input arrays must have the same number of columns!"

        if self.subset and isinstance(x1, pd.DataFrame):
            x1 = x1[self.subset]
            x2 = x2[self.subset]
        
        p = x1.shape[1]
        x1_arr = x1.values if isinstance(x1, pd.DataFrame) else x1
        x2_arr = x2.values if isinstance(x2, pd.DataFrame) else x2
        
        arr_out = pd.Series([
            self.metric(x1_arr[:, col], x2_arr[:, col]) for col in range(p)
        ])
        
        if isinstance(x1, pd.DataFrame):
            arr_out.index = x1.columns
            
        return arr_out
        
    
    def evaluate_1dim_many(self, arrs):
        it = zip(arrs[:-1], arrs[1:])
        arr_out = [self.evaluate_1dim_pair(x1, x2) for x1, x2 in it]
        return arr_out
    
    def evaluate_2dim_many(self, arrs):
        p = arrs[0].shape[1]
        assert all(arr.shape[1] == p for arr in arrs), \
            "Input arrays must have the same number of columns!"

        if self.subset and all(isinstance(arr, pd.DataFrame) for arr in arrs):
            arrs = [arr[self.subset] for arr in arrs]
        
        arrs_np = [
            arr.values if isinstance(arr, pd.DataFrame) else arr
            for arr in arrs
        ]
        
        if self.labels:
            it = zip(
                self.labels[:-1],
                self.labels[1:],
                arrs_np[:-1],
                arrs_np[1:]
            )
            dist_dict = {
                str(lbl1) + self.sep + str(lbl2): self.evaluate_2dim_pair(x1, x2)
                for lbl1, lbl2, x1, x2 in it
            }
        else:
            it = zip(np.arange(p), arrs_np[:-1], arrs_np[1:])
            dist_dict = {lbl: self.evaluate_2dim_pair(x1, x2) for lbl, x1, x2 in it}

        arr_out = pd.DataFrame(dist_dict)
        
        if isinstance(arrs[0], pd.DataFrame):
            arr_out.index = arrs[0].columns
            
        return arr_out
        
    def evaluate(self, *arrs, groupby=None, labels=None):
        if labels:
            self.set_labels(labels)
        if len(arrs) == 2:
            x1, x2 = arrs
            
            if np.array(x1).ndim == 1 and np.array(x2).ndim == 1:
                return self.evaluate_1dim_pair(x1, x2)
                
            elif np.array(x1).ndim == 2 and np.array(x2).ndim == 2:
                return self.evaluate_2dim_pair(x1, x2)
            
            else:
                raise ValueError(
                    "Arrays must be of the same dimension and either 1 or 2 dimensions!"
                )
                
        elif len(arrs) > 2:
            if all(np.array(arr).ndim == 1 for arr in arrs):
                return self.evaluate_1dim_many(arrs)
                
            elif all(np.array(arr).ndim == 2 for arr in arrs):
                return self.evaluate_2dim_many(arrs)
            
            else:
                raise ValueError(
                    "Arrays must be of the same dimension and either 1 or 2 dimensions!"
                )

        elif len(arrs) == 1 and groupby:
            arr = arrs[0]
            grps = np.sort(arr[groupby].unique()).tolist()
            data_split = [df[1].drop(columns=groupby) for df in arr.groupby(groupby)]
            self.set_labels(grps)
            return self.evaluate(*data_split)

    def set_labels(self, labels, sep="_"):
        self.labels = labels
        self.sep = sep

    def set_subset(self, subset):
        self.subset = subset
