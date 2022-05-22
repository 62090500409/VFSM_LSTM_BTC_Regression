import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class WindowGenerator():
    
    def __init__(self, input_width: int, label_width: int, shift: int, 
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 label_columns=None, shuffle=False):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.shuffle = shuffle
        
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = { name: i for i, name in enumerate(label_columns) }
            
        self.column_indices = { name: i for i, name in enumerate(train_df.columns) }
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns ],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def plot(self, plot_col, example=None, model=None, max_subplots=3):
        if example is None:
            inputs, labels = self.example
        else:
            inputs, labels = next(iter(example))
        plt.figure(figsize=(12, 14))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        if len(inputs) > max_subplots:
            step = int(self.input_width / max_n)
        else:
            step = 1
        if model is not None:
            predictions = model(inputs)
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col}')
            plt.plot(self.input_indices, inputs[n*step, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
                
            if label_col_index is None:
                continue
            
            plt.scatter(self.label_indices, labels[n*step, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                plt.scatter(self.label_indices, predictions[n*step, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions', c='#ff7f03', s=64)
                
            if n == 0:
                plt.legend()
        
        plt.xlabel('Date')
        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle,
            batch_size=self.input_width,
        )
        ds = ds.map(self.split_window)
        
        return ds
    
    def __repr__(self):
        return '\n'.join([
            f'Window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result