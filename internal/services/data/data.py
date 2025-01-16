from datetime import datetime
from calendar import monthrange
from tqdm import tqdm
import time

from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import internal.services.data.models as models


def load_candles(years, engine):
  TOKEN = 't.fDimiyDvouBZCRipO5XsUUDuWDv8rRAJwWAZkjytq0KLLwxTKBld4HEXkpE0kC8TrL_Bo1WuValllzW2I-5ddg'
  FIGI = 'BBG004730RP0'

  with Session(engine) as session:
    with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
      for year in years:
        print(f"Start to getting year: {year}")
        for _, month in tqdm(enumerate(range(1, 13))):
          for day in tqdm(range(monthrange(year, month)[1])):
            for candle in client.market_data.get_candles(
                    figi=FIGI,
                    from_=datetime(year, month, day+1, 6, 0),
                    to=datetime(year, month, day+1, 23, 59),
                    interval=CandleInterval.CANDLE_INTERVAL_2_MIN).candles:
              data = models.Candles(
                  figi=FIGI,
                  open=candle.open.units + candle.open.nano / 1000000000,
                  high=candle.high.units + candle.high.nano / 1000000000,
                  low=candle.low.units + candle.low.nano / 1000000000,
                  close=candle.close.units + candle.close.nano / 1000000000,
                  volume=candle.volume,
                  time=candle.time,
                  is_complete=candle.is_complete
              )
              session.add(data)
          time.sleep(0.5)
      session.commit()


if __name__ == '__main__':
  engine = create_engine('sqlite:///storage/sqlite/shares.db')
  engine.connect()
  try:
    models.Candles.__table__.drop(engine)
  except:
    print("[services.data] Table drop error")
  models.Base.metadata.create_all(engine)

  load_candles(range(2024, 2025), engine)
