import pandas as pd

from IPython.display import display, clear_output


def bin_column(df: pd.DataFrame, column_name: str, thresholds: list) -> pd.Series:
    """
    Бинning столбца DataFrame на основе заданных порогов.
    
    Parameters:
        df: Входной DataFrame.
        column_name: Название столбца, который нужно забинировать.
        thresholds: Упорядоченный список пороговых значений.
    
    Returns:
        Забинированный столбец.
    """
    bins = [-float('inf')] + thresholds + [float('inf')]
    binned_column = pd.cut(df[column_name], bins=bins, labels=[i for i in range(len(thresholds) + 1)], include_lowest=True)
    
    return binned_column

def checking(df):
    total = len(df)
    check_df = pd.DataFrame(df.isnull().sum(), columns=['#NULLS'])
    check_df['%NULLS'] = round((check_df['#NULLS']/total)*100, 2)
    check_df['#Unique_Valus'] = df.nunique()
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    uniques = []
    for col in df.columns:
        if col in cat_cols:
            uniques.append(set(df[col].unique()))
        else:
            if df[col].nunique() < 10:
                uniques.append(set(df[col].unique()))
            else:
                uniques.append(df[col].max() - df[col].min())
    check_df['Unique_Values/Range'] = uniques
    return check_df

