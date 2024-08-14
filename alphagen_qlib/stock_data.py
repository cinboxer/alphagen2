from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

import pandas as pd
from sqlalchemy import create_engine, text

"""stock_list = ['SH600009', 'SH600019', 'SH600028', 'SH600030', 'SH600031', 'SH600036',
                'SH600050', 'SH600276', 'SH600309', 'SH600406', 'SH600426', 'SH600436',
                'SH600519', 'SH600585', 'SH600660', 'SH600754', 'SH600887', 'SH600893',
                'SH600900', 'SH601012', 'SH601088', 'SH601318', 'SH601668', 'SH601766',
                'SH601888', 'SH601899', 'SH603259', 'SH603605', 'SH603799', 'SZ000002',
                'SZ000063', 'SZ000333', 'SZ000725', 'SZ000792', 'SZ000938', 'SZ002027',
                'SZ002230', 'SZ002271', 'SZ002352', 'SZ002371', 'SZ002415', 'SZ002466',
                'SZ002594', 'SZ002714', 'SZ300015', 'SZ300122', 'SZ300124', 'SZ300750',
                'SZ300760']"""

def read_index(index_ticker: str, end_date: str):
    import pandas as pd
    from sqlalchemy import create_engine, text

    host=""
    user=""
    password=""
    database=""


    sql_query = """
        SELECT 
            a.SECURITY_ID,
            b.TICKER_SYMBOL AS INDEX_SYMBOL,
            b.SEC_SHORT_NAME AS INDEX_NAME,
            a.CONS_ID,
            c.TICKER_SYMBOL AS STOCK_SYMBOL,
            c.SEC_SHORT_NAME AS STOCK_NAME,
            c.EXCHANGE_CD,
            a.INTO_DATE,
            a.OUT_DATE,
            a.IS_NEW,
            a.UPDATE_TIME
        FROM idx_cons a 
        LEFT JOIN md_security b ON a.SECURITY_ID = b.SECURITY_ID
        LEFT JOIN md_security c ON a.CONS_ID = c.SECURITY_ID
        WHERE b.TICKER_SYMBOL = '"""+ index_ticker +"""' /* 输入需查询的指数代码 */
        AND a.INTO_DATE <= '"""+ end_date +"""' /* 输入需查询的入选日期 */
        AND (a.OUT_DATE IS NULL OR a.OUT_DATE > '"""+ end_date +"""') /* 输入需查询的剔除日期 */
        ORDER BY c.TICKER_SYMBOL;
    """


    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
    df_index = pd.DataFrame(engine.connect().execute(text(sql_query))) 
    df_index.columns = ['SECURITY_ID','INDEX_SYMBOL','INDEX_NAME','CONS_ID','STOCK_SYMBOL','STOCK_NAME','EXCHANGE_CD','INTO_DATE','OUT_DATE','IS_NEW','UPDATE_TIME']
    return df_index


def getdatafromDatayes(cleaned_stock_list, startdate='20200101', enddate='20240601'):
    host=""
    user=""
    password=""
    database=""

    sql_query = """
        SELECT 
            a.TRADE_DATE,
            a.EXCHANGE_CD,
            a.TICKER_SYMBOL,
            a.PRE_CLOSE_PRICE_1,
            a.OPEN_PRICE_1,
            a.CLOSE_PRICE_1,
            a.HIGHEST_PRICE_1,
            a.LOWEST_PRICE_1,
            b.TURNOVER_VOL,
            b.TURNOVER_VALUE,
            a.CLOSE_PRICE
        FROM mkt_equd_adj a
        INNER JOIN mkt_equd b ON (a.TRADE_DATE=b.TRADE_DATE) AND (a.TICKER_SYMBOL=b.TICKER_SYMBOL)
        WHERE (a.TRADE_DATE BETWEEN '""" + startdate + """' and '""" + enddate + """') AND 
        (a.TICKER_SYMBOL IN ("""+ str(cleaned_stock_list).strip('[').strip(']') +"""))
    """

    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
    s_settings_df = pd.DataFrame(engine.connect().execute(text(sql_query)))
    return s_settings_df

def processdata(s_settings_df):
    s_settings_df.columns = ['datetime','EXCHANGE_CD','instrument','preclose','$open','$close','$high','$low','$volume','$value','close_origin']
    s_settings_df = s_settings_df.dropna()
    s_settings_df = s_settings_df[(s_settings_df['$value']!=0) & (s_settings_df['$volume']!=0)]
    s_settings_df['$vwap'] = s_settings_df['$value'] / s_settings_df['$volume'] / s_settings_df['close_origin'] * s_settings_df['$close']

    exchange_map = {'XSHE': 'SZ', 'XBEI': 'BJ', 'XSHG': 'SH'}
    #s_settings_df['instrument'] = exchange_map[s_settings_df['EXCHANGE_CD'].iloc[0]] + s_settings_df['instrument']
    s_settings_df['instrument'] = s_settings_df['EXCHANGE_CD'].map(exchange_map) + s_settings_df['instrument']
    s_settings_df = s_settings_df[s_settings_df['EXCHANGE_CD']!='XBEI']
    s_settings_df.drop(columns=['EXCHANGE_CD', 'preclose', '$value', 'close_origin'], inplace=True)
    s_settings_df = s_settings_df.sort_values(by=['datetime', 'instrument'], ascending=True)
    multi_index_df = s_settings_df.set_index(['datetime', 'instrument'])

    multi_index_df = multi_index_df.astype('float32')
    return multi_index_df


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_rolling", region=REG_CN)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[start_index - self.max_backtrack_days]
        if cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_time = cal[end_index + self.max_future_days]

        # Get the index data
        index2ticker = {'csi300': '000300', 'csi500': '000905', 'zza50': '930050'}
        df_index = read_index(index2ticker[self._instrument], '2024-08-08') # zza50
        exchange_map = {'XSHE': 'SZ', 'XBEI': 'BJ', 'XSHG': 'SH'}
        df_index['STOCK_SYMBOL'] = df_index['EXCHANGE_CD'].map(exchange_map) + df_index['STOCK_SYMBOL']
        stock_list = list(df_index['STOCK_SYMBOL'])
        cleaned_stock_list = [stock[2:] for stock in stock_list]
        return processdata(getdatafromDatayes(cleaned_stock_list, startdate=real_start_time.strftime('%Y%m%d'), enddate=real_end_time.strftime('%Y%m%d')))

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = df.stack().unstack(level=1)
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
