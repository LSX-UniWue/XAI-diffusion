import numpy as np


def get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized', add_is_normal=False, add_attack_type=False):
    if cat_encoding == 'ordinal' and num_encoding == 'zscore':
        cols = ['isWeekday', 'Daytime', 'Duration', 'isICMP', 'isUDP', 'isTCP', 'Src IP', 'Src Pt', 'Src Conns',
                'Dst IP', 'Dst Pt', 'Dst Conns', 'Packets', 'Bytes', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
        dtypes = {
            **{col: np.int8 for col in ['isWeekday', 'isICMP', 'isUDP', 'isTCP', 'isSYN', 'isACK', 'isFIN', 'isURG',
                                        'isPSH', 'isRES', 'Src IP', 'Src Pt', 'Dst IP', 'Dst Pt']},
            **{col: np.float32 for col in ['Daytime', 'Duration', 'Packets', 'Bytes', 'Src Conns', 'Dst Conns']}
        }

    if cat_encoding == 'onehot' and num_encoding == 'minmax':
        cols = ['isWeekday', 'Daytime', 'Duration', 'isICMP', 'isUDP', 'isTCP',
                *[f'Src IP {x}' for x in range(36)], *[f'Src Pt {x}' for x in range(19)], 'Src Conns',
                *[f'Dst IP {x}' for x in range(36)], *[f'Dst Pt {x}' for x in range(19)], 'Dst Conns',
                'Packets', 'Bytes', 'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
        dtypes = {**{col: np.int8 for col in ['isWeekday', 'isICMP', 'isUDP', 'isTCP', 'isSYN', 'isACK', 'isFIN',
                                              'isURG', 'isPSH', 'isRES',
                                              *[f'Src IP {x}' for x in range(36)], *[f'Src Pt {x}' for x in range(19)],
                                              *[f'Dst IP {x}' for x in range(36)],
                                              *[f'Dst Pt {x}' for x in range(19)]]},
                  **{col: np.float32 for col in ['Daytime', 'Duration', 'Packets', 'Bytes', 'Src Conns', 'Dst Conns']}}

    if cat_encoding == 'onehot' and num_encoding == 'quantized':
        cols = ['isWeekday', *[f'Daytime {x}' for x in range(5)], *[f'Duration {x}' for x in range(4)], 'isICMP',
                'isUDP', 'isTCP',
                *[f'Src IP {x}' for x in range(36)], *[f'Src Pt {x}' for x in range(19)],
                *[f'Src Conns {x}' for x in range(4)],
                *[f'Dst IP {x}' for x in range(36)], *[f'Dst Pt {x}' for x in range(19)],
                *[f'Dst Conns {x}' for x in range(4)],
                *[f'Packets {x}' for x in range(4)], *[f'Bytes {x}' for x in range(5)],
                'isSYN', 'isACK', 'isFIN', 'isURG', 'isPSH', 'isRES']
        dtypes = {col_name: np.int8 for col_name in cols}

    if add_is_normal:
        cols.append('isNormal')
        dtypes['isNormal'] = np.int8

    if add_attack_type:
        cols.append('attackType')
        dtypes['attackType'] = np.int8

    return cols, dtypes


def get_column_mapping(cat_encoding='onehot', num_encoding='quantized', as_int=True):
    """Column mapping that lists which columns encode the same information"""
    if cat_encoding == 'onehot' and num_encoding == 'quantized':
        if as_int:
            cols = [[0],
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9],
                    [10, 11, 12],
                    list(range(13, 13 + 36)),
                    list(range(49, 49 + 19)),
                    [68, 69, 70, 71],
                    list(range(72, 72 + 36)),
                    list(range(108, 108 + 19)),
                    [127, 128, 129, 130],
                    [131, 132, 133, 134],
                    [135, 136, 137, 138, 139],
                    [140],
                    [141],
                    [142],
                    [143],
                    [144],
                    [145]]
        else:
            cols = [['isWeekday'],
                    [*[f'Daytime {x}' for x in range(5)]],
                    [*[f'Duration {x}' for x in range(4)]],
                    ['isICMP', 'isUDP', 'isTCP'],
                    [*[f'Src IP {x}' for x in range(36)]],
                    [*[f'Src Pt {x}' for x in range(19)]],
                    [*[f'Src Conns {x}' for x in range(4)]],
                    [*[f'Dst IP {x}' for x in range(36)]],
                    [*[f'Dst Pt {x}' for x in range(19)]],
                    [*[f'Dst Conns {x}' for x in range(4)]],
                    [*[f'Packets {x}' for x in range(4)]],
                    [*[f'Bytes {x}' for x in range(5)]],
                    ['isSYN'],
                    ['isACK'],
                    ['isFIN'],
                    ['isURG'],
                    ['isPSH'],
                    ['isRES']]
    else:
        raise NotImplementedError

    return cols


def map_orig_columns(cols):
    cols = [col.strip() for col in cols]
    map_dict = {'Date first seen': 'Date',
                'Proto': 'Traffic',
                'Src IP Addr': 'Src IP',
                'Dst IP Addr': 'Dst IP',
                'Dst Connections last 10s': 'Dst Conns',
                'Src Connections last 10s': 'Src Conns'}
    for i, col in enumerate(cols):
        if col in map_dict:
            cols[i] = map_dict[col]
    return cols


def get_summed_columns():
    cols = ['Weekday', 'Daytime', 'Duration', 'Traffic', 'Src IP', 'Src Pt', 'Src Conns',
            'Dst IP', 'Dst Pt', 'Dst Conns', 'Packets', 'Bytes', 'SYN', 'ACK', 'FIN', 'URG', 'PSH', 'RES']
    return cols
