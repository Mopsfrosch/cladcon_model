'''Adapted from https://github.com/pfnet-research/head_model/tree/master'''


def use_hsa_mirna_only(df):
    return df[df['G_Name'].str.startswith('hsa-')]
